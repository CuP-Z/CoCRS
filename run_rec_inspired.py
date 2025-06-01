#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The standard way to train a model. After training, also computes validation
and test error.

The user must provide a model (with ``--model``) and a task (with ``--task`` or
``--pytorch-teacher-task``).

Examples
--------

.. code-block:: shell

  python -m parlai.scripts.train -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
  python -m parlai.scripts.train -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128
  python -m parlai.scripts.train -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

"""  # noqa: E501

# TODO List:
# * More logging (e.g. to files), make things prettier.

import numpy as np
from tqdm import tqdm
from math import exp
import os
DEVICES = "1"
os.environ['CUDA_VISIBLE_DEVICES']=DEVICES
import signal
import json
import argparse
import pickle as pkl
from dataset_rec_inspired import dataset,CRSdataset
from model_rec_inspired import CrossModel
import torch.nn as nn
from torch import optim
from loguru import logger
import torch
try:
    import torch.version
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu
import sys
import time
import random
# data_type = "redial"
data_type = "inspired"

seed = 55555
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

logger.remove()
local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
logger.add(sys.stderr, level='DEBUG')
logger.add(f'log_rec/{data_type}+{local_time}.log', level='DEBUG')
logger.info("DEVICES: "+DEVICES)
def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()

def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-max_turn","--max_turn",type=str,default=0)
    train.add_argument("-dataset","--dataset",type=str,default=data_type)
    train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    train.add_argument("-max_r_length","--max_r_length",type=int,default=30)
    train.add_argument("-batch_size","--batch_size",type=int,default=512)
    train.add_argument("-max_count","--max_count",type=int,default=5)# 5
    # train.add_argument("-max_count","--max_count",type=int,default=4)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True)
    train.add_argument("-load_dict","--load_dict",type=str,default=None)
    train.add_argument("-learningrate","--learningrate",type=float,default=1e-3)#1e-3
    train.add_argument("-optimizer","--optimizer",type=str,default='adam')
    train.add_argument("-momentum","--momentum",type=float,default=0)
    train.add_argument("-is_finetune","--is_finetune",type=bool,default=False)
    train.add_argument("-embedding_type","--embedding_type",type=str,default='random')
    train.add_argument("-epoch","--epoch",type=int,default=20)
    train.add_argument("-gpu","--gpu",type=str,default='0,1')
    # train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)#0.1 redial #0.5
    train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.022)#inspired
    train.add_argument("-embedding_size","--embedding_size",type=int,default=128)#128 #100
    train.add_argument("-dim","--dim",type=int,default=128) #128

# train.add_argument("-n_heads","--n_heads",type=int,default=2)
    # train.add_argument("-n_layers","--n_layers",type=int,default=2)
    # train.add_argument("-ffn_size","--ffn_size",type=int,default=300)

    train.add_argument("-dropout","--dropout",type=float,default=0.1)
    train.add_argument("-attention_dropout","--attention_dropout",type=float,default=0.0)
    train.add_argument("-relu_dropout","--relu_dropout",type=float,default=0.1)

    train.add_argument("-learn_positional_embeddings","--learn_positional_embeddings",type=bool,default=False)
    train.add_argument("-embeddings_scale","--embeddings_scale",type=bool,default=True)

    # train.add_argument("-n_entity","--n_entity",type=int,default=64368) #redial
    # train.add_argument("-n_concept","--n_concept",type=int,default=29308) #redial
    
    train.add_argument("-n_concept","--n_concept",type=int,default=56490) #inspired
    train.add_argument("-n_entity","--n_entity",type=int,default=17321) #inspired
    train.add_argument("-n_relation","--n_relation",type=int,default=214)

    # train.add_argument("-n_con_relation","--n_con_relation",type=int,default=48)
    train.add_argument("-n_hop","--n_hop",type=int,default=2)
    train.add_argument("-kge_weight","--kge_weight",type=float,default=1)
    train.add_argument("-l2_weight","--l2_weight",type=float,default=2.5e-6)
    train.add_argument("-n_memory","--n_memory",type=float,default=32)
    train.add_argument("-item_update_mode","--item_update_mode",type=str,default='0,1')
    train.add_argument("-using_all_hops","--using_all_hops",type=bool,default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)

    return train

class TrainLoop_fusion_rec():
    def __init__(self, opt, is_finetune):
        logger.info(opt)
        logger.info("turn : "+str(opt['max_turn']))
        self.opt=opt
        self.train_dataset=dataset('data-{}/train_data.jsonl'.format(opt['dataset']),opt,max_turn=opt['max_turn'])
        self.val_dataset = dataset('data-{}/test_data.jsonl'.format(self.opt['dataset']), self.opt,max_turn=opt['max_turn'])

        self.dict=self.train_dataset.word2index
        self.index2word={self.dict[key]:key for key in self.dict}
        self.rec_results = []
        self.rec_labels = []
        
        self.context_sum_label=[]
        self.responce_sum_label = []
        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']
        if opt['load_dict']!=None:
            self.load_data=True
        else:
            self.load_data=False
        self.is_finetune=False

        self.movie_ids = pkl.load(open("data-{}/movie_ids.pkl".format(opt['dataset']), "rb"))
        # self.movie_ids = json.load(open("data-{}/movie_ids.json".format(opt['dataset']),'rb'))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune)

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

    def build_model(self,is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune=is_finetune,tokenizer=self.train_dataset.bert_tokenizer)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        #self.model.load_model()
        losses=[]
        best_val_rec=0
        # train_dataset = dataset('data-{}/train_data_dbpedia_raw.jsonl'.format(self.opt['dataset']),self.opt)
        # train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        # train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        rec_stop=False
        output_metrics_rec = self.val(is_test=True)
        self.save_result()
        pre_train_dataset=dataset('data-{}/train_data.jsonl'.format(self.opt['dataset']),self.opt,max_turn=self.opt['max_turn'])
        pre_train_set=CRSdataset(pre_train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        train_dataset_loader = torch.utils.data.DataLoader(dataset=pre_train_set,batch_size=self.batch_size,shuffle=True)
        info_cur_h_loss_mu = 0.8 
        # info_cur_h_loss_mu = 0.5
        # info_cur_h_loss_mu = 0.8
        logger.info("info_cur_h_loss_mu: "+str(info_cur_h_loss_mu))
        for i in range(3):
            num=0
            # for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec in tqdm(train_dataset_loader):
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,context_raw,c_raw_length,response_raw,r_raw_length,overview,o_length,simi_user,simi_user_vector,simi_user_entity_vec,cur_entity_vec in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_set_simi = simi_user_entity_vec[b].nonzero().view(-1).tolist()
                    # seed_set_simi = []
                    seed_sets.append(seed_set+seed_set_simi)
                
                # simi_user_seed_sets = []
                # for b in range(batch_size):
                #     seed_set = simi_user_entity_vec[b].nonzero().view(-1).tolist()
                #     simi_user_seed_sets.append(seed_set)

                cur_entity_vec_seed_sets = []
                for b in range(batch_size):
                    seed_set = cur_entity_vec[b].nonzero().view(-1).tolist()
                    cur_entity_vec_seed_sets.append(seed_set)

                # seed_sets = seed_sets+simi_user_seed_sets
                simi_user_sets = []
                for b in range(batch_size):
                    seed_set = simi_user[b].nonzero().view(-1).tolist()
                    simi_user_sets.append(seed_set)
                self.model.train()
                self.zero_grad()
                # print('seed_set_simi',len(seed_set_simi))
                # print('simi_user_sets',len(seed_set))

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_cur_h_loss=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,concept_vec, db_vec, entity_vector.cuda(), rec,simi_user_sets,context_raw.cuda(),cur_entity_vec_seed_sets, test=False)
                # scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, _=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,concept_vec, db_vec, entity_vector.cuda(), rec,simi_user_sets,context_raw.cuda(), test=False)
                
                # joint_loss=0.5*rec_loss+0.5*info_db_loss#+0.0*info_con_loss#+mask_loss*0.05
                # losses.append([rec_loss,info_db_loss])
                # self.backward(joint_loss)

                # joint_loss=info_db_loss + 0.2 * info_cur_h_loss #+info_con_loss
                # joint_loss=info_db_loss + 0.1 * info_cur_h_loss #+info_con_loss
                # joint_loss=info_db_loss + 0.05 * info_cur_h_loss #+info_con_loss
                # joint_loss=info_db_loss + 0.5 * info_cur_h_loss #+info_con_loss
                joint_loss=info_db_loss + info_cur_h_loss_mu * info_cur_h_loss #+info_con_loss

                losses.append([info_db_loss])
                self.backward(joint_loss)

                self.update_params()
                if num%50==0:
                    print('info db loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    #print('info con loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    losses=[]
                num+=1

        print("masked loss pre-trained")
        losses=[]

        train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=self.batch_size,shuffle=True)

        # lambda1,lambda2 = 0.025,0.001
        lambda1,lambda2 = 0.008,0.001
        # lambda1,lambda2 = 0.01,0.002
        # lambda1,lambda2 = 0.5,0.05
        logger.info("lambda1: "+str(lambda1))
        logger.info("lambda2: "+str(lambda2))

        for i in range(self.epoch):
            # train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
            #                                                 batch_size=self.batch_size,
            #                                                 shuffle=True)
            num=0
            # for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec in tqdm(train_dataset_loader):
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,context_raw,c_raw_length,response_raw,r_raw_length,overview,o_length,simi_user,simi_user_vector,simi_user_entity_vec,cur_entity_vec in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_set_simi = simi_user_entity_vec[b].nonzero().view(-1).tolist()
                    # seed_set_simi = []
                    seed_sets.append(seed_set+seed_set_simi)

                cur_entity_vec_seed_sets = []
                for b in range(batch_size):
                    seed_set = cur_entity_vec[b].nonzero().view(-1).tolist()
                    cur_entity_vec_seed_sets.append(seed_set)

                # seed_sets = seed_sets+simi_user_seed_sets

                simi_user_sets = []
                for b in range(batch_size):
                    seed_set = simi_user[b].nonzero().view(-1).tolist()
                    simi_user_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                # scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, _=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,concept_vec, db_vec, entity_vector.cuda(), rec,simi_user_sets, test=False)
                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_cur_h_loss=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,concept_vec, db_vec, entity_vector.cuda(), rec,simi_user_sets,context_raw.cuda(),cur_entity_vec_seed_sets, test=False)

                # joint_loss=rec_loss+0.008*info_db_loss#+0.0*info_con_loss#+mask_loss*0.05
                # joint_loss=rec_loss+0.025*info_db_loss#+0.0*info_con_loss#+mask_loss*0.05
                # joint_loss=rec_loss+0.02*info_db_loss + 0.002*info_cur_h_loss#+0.0*info_con_loss#+mask_loss*0.05
                joint_loss=rec_loss+lambda1*info_db_loss + lambda2*info_cur_h_loss#+0.0*info_con_loss#+mask_loss*0.05

                losses.append([rec_loss,info_db_loss,info_cur_h_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%50==0:
                    print('rec loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    print('info db loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    print('info cur loss is %f'%(sum([l[2] for l in losses])/len(losses)))
                    losses=[]
                num+=1

            output_metrics_rec = self.val(is_test=True)

            if best_val_rec > output_metrics_rec["recall@10"]+output_metrics_rec["recall@1"]:
                rec_stop=True
            else:
                best_val_rec = output_metrics_rec["recall@10"]+output_metrics_rec["recall@1"]
                # self.model.save_model()
                self.save_result()
                print("recommendation model saved once------------------------------------------------")

            # if rec_stop==True:
            #     break

        _=self.val(is_test=True)
    def save_result(self):
        temp_result = []
        for i in range(len(self.rec_labels)):
            dic = {}
            response = ""
            context = ""
            for word in self.responce_sum_label[i]:
                response += word +" "
            for line in self.context_sum_label[i]:
                # for word in line:
                #     context += word +" "
                context += " "+ line +  " "
            dic['rec_label'] = self.rec_labels[i]
            dic['rec_pred_50'] = self.rec_results[i]
            # dic['context'] = self.context_sum_label[i]
            # dic['context'] = [' '.join(sen)+'\n' for sen in self.context_sum_label[i]]
            dic['response'] = response
            dic['context'] = context
            # dic['response'] = [' '.join(sen)+'\n' for sen in self.responce_sum_label[i]]
            # dic['response'] = self.responce_sum_label[i]
            temp_result.append(dic)
        
        import jsonlines
        import json
        with open("rec_results_{}.jsonl".format(self.opt['dataset']), "w", encoding="utf-8") as f:
            for i in temp_result:
                json.dump(i,f)
                f.write('\n')
        
    def vector2sentence(self,batch_sen):
        sentences=[]
        for sen in batch_sen.numpy().tolist():
            sentence=[]
            for word in sen:
                if word>3:
                    sentence.append(self.index2word[word])
                elif word==3:
                    sentence.append('_UNK_')
            sentences.append(sentence)
        return sentences
    
    def metrics_cal_rec(self,rec_loss,scores,labels,save=False):
        import math
        def compute_mrr(label,rank, k):
            if label in rank[:k]:
                label_rank = rank.index(label)
                return 1 / (label_rank + 1)
            return 0
        def compute_ndcg(label,rank, k):
            if label in rank[:k]:
                label_rank = rank.index(label)
                return 1 / math.log2(label_rank + 2)
            return 0
        batch_size = len(labels.view(-1).tolist())
        self.metrics_rec["loss"] += rec_loss
        outputs = scores.cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=50, dim=1)
        for b in range(batch_size):
            if labels[b].item()==0:
                continue
            target_idx = self.movie_ids.index(labels[b].item())
            self.rec_labels.append(labels[b].item())
            temp_rec = []
            for id in pred_idx[b][:50].tolist():
                temp_rec.append(self.movie_ids[id])
            self.rec_results.append(temp_rec)
            self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.metrics_rec["count"] += 1

            ##ndcg
            self.metrics_rec["ndcg@1"] += compute_ndcg(target_idx, pred_idx[b].tolist(), 1)
            self.metrics_rec["ndcg@10"] += compute_ndcg(target_idx, pred_idx[b].tolist(), 10)
            self.metrics_rec["ndcg@50"] += compute_ndcg(target_idx, pred_idx[b].tolist(), 50)
            ##mrr
            self.metrics_rec["mrr@1"] += compute_mrr(target_idx, pred_idx[b].tolist(), 1)
            self.metrics_rec["mrr@10"] += compute_mrr(target_idx, pred_idx[b].tolist(), 10)
            self.metrics_rec["mrr@50"] += compute_mrr(target_idx, pred_idx[b].tolist(), 50)


    def val(self,is_test=False,save=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        # self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"ndcg@1":0,"ndcg@10":0,"ndcg@50":0,"mrr@1":0,"mrr@10":0,"mrr@50":0,
        "loss":0,"gate":0,"count":0,'gate_count':0}
        self.rec_labels = []
        self.rec_results = []
        self.context_sum_label = []
        self.responce_sum_label = []
        self.model.eval()
        # if is_test:
        #     val_dataset = dataset('data-{}/test_data.jsonl'.format(self.opt['dataset']), self.opt,pre_train=False)
        # else:
        #     val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set=CRSdataset(self.val_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        recs=[]
        # for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(val_dataset_loader):
        # for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,context_raw,c_raw_length,response_raw,r_raw_length,overview,o_length,simi_user,simi_user_vector in tqdm(val_dataset_loader):
        for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,context_raw,c_raw_length,response_raw,r_raw_length,overview,o_length,simi_user,simi_user_vector,simi_user_entity_vec,cur_entity_vec in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_set_simi = simi_user_entity_vec[b].nonzero().view(-1).tolist()
                    # seed_set_simi = []
                    seed_sets.append(seed_set+seed_set_simi)
                
                # simi_user_seed_sets = []
                # for b in range(batch_size):
                #     seed_set = simi_user_entity_vec[b].nonzero().view(-1).tolist()
                #     simi_user_seed_sets.append(seed_set)

                cur_entity_vec_seed_sets = []
                for b in range(batch_size):
                    seed_set = cur_entity_vec[b].nonzero().view(-1).tolist()
                    cur_entity_vec_seed_sets.append(seed_set)

                # seed_sets = seed_sets+simi_user_seed_sets

                simi_user_sets = []
                for b in range(batch_size):
                    seed_set = simi_user[b].nonzero().view(-1).tolist()
                    simi_user_sets.append(seed_set)
                # scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec,simi_user_sets,context_raw.cuda(), test=False, maxlen=20, bsz=batch_size)
                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, _=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,concept_vec, db_vec, entity_vector.cuda(), rec,simi_user_sets,context_raw.cuda(),cur_entity_vec_seed_sets, test=False)

            recs.extend(rec.cpu())
            #print(losses)
            #exit()
            self.responce_sum_label.extend(self.vector2sentence(response.cpu()))
            self.context_sum_label.extend(self.vector2sentence(context.cpu()))
            
            self.metrics_cal_rec(rec_loss, rec_scores, movie)

        output_dict_rec={key: self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        # print(output_dict_rec)
        logger.info(output_dict_rec)

        return output_dict_rec

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

if __name__ == '__main__':
    args=setup_args().parse_args()
    print(vars(args))
    loop=TrainLoop_fusion_rec(vars(args),is_finetune=False)
    loop.train()
    met=loop.val(True)
