
import numpy as np
from tqdm import tqdm
from math import exp
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import signal
import json
import argparse
from transformers import AutoTokenizer, AutoModel,BertForPreTraining

import pickle as pkl
from dataset_con import dataset,CRSdataset
# from model_copy import CrossModel
from model_con import CrossModel
import torch.nn as nn
from torch import optim
import torch
try:
    import torch.version
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu
import random

# seed = 55555
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# data_type = "redial"
data_type = "inspired"

import sys
import time
from loguru import logger
local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
logger.remove()
logger.add(sys.stderr, level='DEBUG')
logger.add(f'log_gen/{data_type}-{local_time}.log', level='DEBUG')

def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()

def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-dataset","--dataset",type=str,default=data_type)
    train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    train.add_argument("-max_r_length","--max_r_length",type=int,default=256)
    # train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    # train.add_argument("-max_r_length","--max_r_length",type=int,default=30)
    # train.add_argument("-max_gen_len","--max_gen_len",type=int,default=50) #inspired 
    train.add_argument("-max_gen_len","--max_gen_len",type=int,default=30) #redial  30
    train.add_argument("-batch_size","--batch_size",type=int,default=11)
    # train.add_argument("-max_count","--max_count",type=int,default=5)
    train.add_argument("-max_count","--max_count",type=int,default=10)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True)
    train.add_argument("-load_dict","--load_dict",type=str,default=None)
    # train.add_argument("-learningrate","--learningrate",type=float,default=1e-3)
    train.add_argument("-learningrate","--learningrate",type=float,default=1e-5)
    train.add_argument("-optimizer","--optimizer",type=str,default='adam')
    train.add_argument("-momentum","--momentum",type=float,default=0)
    train.add_argument("-is_finetune","--is_finetune",type=bool,default=True)
    train.add_argument("-embedding_type","--embedding_type",type=str,default='random')
    train.add_argument("-epoch","--epoch",type=int,default=90)
    train.add_argument("-gpu","--gpu",type=str,default='0,1')
    train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1) #0.1 0.22
    train.add_argument("-embedding_size","--embedding_size",type=int,default=300)

    train.add_argument("-n_heads","--n_heads",type=int,default=2)
    train.add_argument("-n_layers","--n_layers",type=int,default=2)
    train.add_argument("-ffn_size","--ffn_size",type=int,default=300)

    train.add_argument("-dropout","--dropout",type=float,default=0.1)
    train.add_argument("-attention_dropout","--attention_dropout",type=float,default=0.0)
    train.add_argument("-relu_dropout","--relu_dropout",type=float,default=0.1)

    train.add_argument("-learn_positional_embeddings","--learn_positional_embeddings",type=bool,default=False)
    train.add_argument("-embeddings_scale","--embeddings_scale",type=bool,default=True)

    train.add_argument("-n_concept","--n_concept",type=int,default=56490) #inspired
    train.add_argument("-n_entity","--n_entity",type=int,default=17321) #inspired

    # train.add_argument("-n_entity","--n_entity",type=int,default=64368)
    # train.add_argument("-n_entity","--n_entity",type=int,default=64363)
    train.add_argument("-n_relation","--n_relation",type=int,default=214)
    # train.add_argument("-n_concept","--n_concept",type=int,default=29308)
    train.add_argument("-n_con_relation","--n_con_relation",type=int,default=48)
    train.add_argument("-dim","--dim",type=int,default=128)
    train.add_argument("-n_hop","--n_hop",type=int,default=2)
    train.add_argument("-kge_weight","--kge_weight",type=float,default=1)
    train.add_argument("-l2_weight","--l2_weight",type=float,default=2.5e-6)
    train.add_argument("-n_memory","--n_memory",type=float,default=32)
    train.add_argument("-item_update_mode","--item_update_mode",type=str,default='0,1')
    train.add_argument("-using_all_hops","--using_all_hops",type=bool,default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)

    return train


class TrainLoop_fusion_gen():
    def __init__(self, opt, is_finetune):
        self.context_sum = []
        self.inference_sum = []
        self.golden_sum = []
        self.i = 0
        self.opt=opt
        self.tokenizer = AutoTokenizer.from_pretrained('/home/zhy/crs/MyCRS-newdata/DialoGPT-small')
        gpt2_special_tokens_dict = {
            'pad_token': '<pad>',
            'additional_special_tokens': ['<movie>'],
        }
        self.tokenizer.add_special_tokens(gpt2_special_tokens_dict)

        # self.test_dataset=dataset('data-{}/test_data_dbpedia_raw.jsonl'.format(opt['dataset']),opt,self.tokenizer)
        self.test_dataset=dataset('data-{}/test_data.jsonl'.format(opt['dataset']),opt,self.tokenizer)
        self.test_set=CRSdataset(self.test_dataset.data_process(True),self.test_dataset.entity_num,self.opt['n_concept'],self.opt)
        # self.train_dataset=dataset('data-{}/train_data_dbpedia_raw.jsonl'.format(opt['dataset']),opt,self.tokenizer)
        self.train_dataset=dataset('data-{}/train_data.jsonl'.format(opt['dataset']),opt,self.tokenizer)

        self.dict=self.train_dataset.word2index
        self.index2word={self.dict[key]:key for key in self.dict}

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']
        if opt['load_dict']!=None:
            self.load_data=True
        else:
            self.load_data=False
        self.is_finetune=False

        self.movie_ids = pkl.load(open("data-{}/movie_ids.pkl".format(opt['dataset']), "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune=True)

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
        self.model = CrossModel(self.opt, self.dict, is_finetune=is_finetune,tokenizer=self.tokenizer)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        _=self.val(is_test=True)
        # self.model.load_model()
        losses=[]
        best_val_gen=1000
        gen_stop=False
        train_set=CRSdataset(self.train_dataset.data_process(True),self.train_dataset.entity_num,self.opt['n_concept'],self.opt)
        train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)
        for i in range(self.epoch):
            num=0
            self.i = i+1
            for context,context_mask,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,context_raw,c_raw_length,response_raw,r_raw_length,overview,o_length,simi_user,simi_user_vector in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                simi_user_sets = []
                for b in range(batch_size):
                    seed_set = simi_user[b].nonzero().view(-1).tolist()
                    simi_user_sets.append(seed_set)
                
                # continue
                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss=self.model(context.cuda(),context_mask.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,concept_vec, db_vec, entity_vector.cuda(), rec,context_raw.cuda(),response_raw.cuda(),overview.cuda(),simi_user_sets,simi_user_vector.cuda(), test=False,conv=True)

                joint_loss=gen_loss

                losses.append([gen_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%50==0:
                    print('gen loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    losses=[]
                num+=1
            if i%1==0:
                output_metrics_gen = self.val(True)
                if best_val_gen < output_metrics_gen["bleu2"]:
                    pass
                else:
                    best_val_gen = output_metrics_gen["bleu2"]
                    # self.model.save_model()
                    f=open('context_test_{}_best.txt'.format(self.opt['dataset']),'w',encoding='utf-8')
                    f.writelines([' '.join(sen)+'\n' for sen in self.context_sum])
                    f.close()

                    f=open('output_test_{}_best.txt'.format(self.opt['dataset']),'w',encoding='utf-8')
                    f.writelines([' '.join(sen)+'\n' for sen in self.inference_sum])
                    f.close()
                    
                    f=open('output_label_{}_best.txt'.format(self.opt['dataset']),'w',encoding='utf-8')
                    f.writelines([' '.join(sen)+'\n' for sen in self.golden_sum])
                    f.close()
                    # print("generator model saved once------------------------------------------------")
                    print("generator response saved once------------------------------------------------")

        _=self.val(is_test=True)

    def val(self,is_test=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        self.model.eval()
        # val_dataset = dataset('data-{}/test_data_dbpedia_raw.jsonl'.format(self.opt['dataset']), self.opt,self.tokenizer,gen=True)
        val_dataset = dataset('data-{}/test_data.jsonl'.format(self.opt['dataset']), self.opt,self.tokenizer,gen=True)
        # if is_test:
        # else:
        #     val_dataset = dataset('data-{}/valid_data.jsonl'.format(self.opt['dataset']), self.opt,self.tokenizer,gen=True)
        val_set=CRSdataset(val_dataset.data_process(True),val_dataset.entity_num,self.opt['n_concept'],self.opt)
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        inference_sum=[]
        golden_sum=[]
        context_sum=[]
        losses=[]
        recs=[]
        for context,context_mask,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,context_raw,c_raw_length,response_raw,r_raw_length,overview,o_length,simi_user,simi_user_vector in tqdm(val_dataset_loader):
            with torch.no_grad():
                # seed_sets = []
                # batch_size = context.shape[0]
                # for b in range(batch_size):
                #     seed_set = entity[b].nonzero().view(-1).tolist()
                #     seed_sets.append(seed_set)
                # scores, preds, _, _, gen_loss, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(),context_mask.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=False,conv=True)
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                simi_user_sets = []
                for b in range(batch_size):
                    seed_set = simi_user[b].nonzero().view(-1).tolist()
                    simi_user_sets.append(seed_set)
                
                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss=self.model(context.cuda(),context_mask.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,concept_vec, db_vec, entity_vector.cuda(), rec,context_raw.cuda(),response_raw.cuda(),overview.cuda(),simi_user_sets,simi_user_vector.cuda(), test=False,conv=True)
                
                input_dic = {
                    "input_ids":context.cuda(),
                    "attention_mask":context_mask.cuda(),
                }
                gen_seqs = self.model.transformer.generate(
                    **input_dic,
                    max_new_tokens=args.max_gen_len,
                    # no_repeat_ngram_size=3,
                    no_repeat_ngram_size=5,
                )
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, context_mask):
                    gen_seq = [token_id for token_id in gen_seq if token_id != self.tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[sum(length):])
                # import pdb
                # pdb.set_trace()
                # scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(), context_mask.cuda(),response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=True, maxlen=20, bsz=batch_size)
            label = [res[2:lenth] for res,lenth in zip(response.cpu(),r_length.cpu())]
            contexts = [con[-sum(con_len):] for con,con_len,lenth in zip(context.cpu(),context_mask.cpu(),r_length.cpu())]
            golden_sum.extend(self.vector2sentence(label))
            inference_sum.extend(self.vector2sentence(gen_resp_ids))
            context_sum.extend(self.vector2sentence(contexts,is_context=True))
            recs.extend(rec.cpu())
            losses.append(torch.mean(gen_loss))
            #print(losses)
            #exit()

        self.metrics_cal_gen(losses,inference_sum,golden_sum,recs)

        output_dict_gen={}
        for key in self.metrics_gen:
            if 'bleu' in key:
                output_dict_gen[key]=self.metrics_gen[key]/self.metrics_gen['count']
            else:
                output_dict_gen[key]=self.metrics_gen[key]
        # print(output_dict_gen)
        logger.info(output_dict_gen)
        f=open('./_generation/context_test_{}_{}.txt'.format(self.opt['dataset'],self.i),'w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in context_sum])
        f.close()

        f=open('./_generation/output_test_{}_{}.txt'.format(self.opt['dataset'],self.i),'w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in inference_sum])
        f.close()
        
        f=open('./_generation/output_label_{}_{}.txt'.format(self.opt['dataset'],self.i),'w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in golden_sum])
        f.close()
        
        return output_dict_gen

    def metrics_cal_gen(self,rec_loss,preds,responses,recs):
        def bleu_cal(sen1, tar1):
            bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
            bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
            bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
            return bleu1, bleu2, bleu3, bleu4

        def distinct_metrics(outs):
            # outputs is a list which contains several sentences, each sentence contains several words
            unigram_count = 0
            bigram_count = 0
            trigram_count=0
            quagram_count=0
            unigram_set = set()
            bigram_set = set()
            trigram_set=set()
            quagram_set=set()
            for sen in outs:
                for word in sen:
                    unigram_count += 1
                    unigram_set.add(word)
                for start in range(len(sen) - 1):
                    bg = str(sen[start]) + ' ' + str(sen[start + 1])
                    bigram_count += 1
                    bigram_set.add(bg)
                for start in range(len(sen)-2):
                    trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                    trigram_count+=1
                    trigram_set.add(trg)
                for start in range(len(sen)-3):
                    quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                    quagram_count+=1
                    quagram_set.add(quag)
            dis1 = len(unigram_set) / len(outs)#unigram_count
            dis2 = len(bigram_set) / len(outs)#bigram_count
            dis3 = len(trigram_set)/len(outs)#trigram_count
            dis4 = len(quagram_set)/len(outs)#quagram_count
            return dis1, dis2, dis3, dis4

        predict_s=preds
        golden_s=responses
        #print(rec_loss[0])
        self.metrics_gen["ppl"]+=sum([exp(ppl) for ppl in rec_loss])/len(rec_loss)
        generated=[]

        for out, tar, rec in zip(predict_s, golden_s, recs):
            bleu1, bleu2, bleu3, bleu4=bleu_cal(out, tar)
            generated.append(out)
            self.metrics_gen['bleu1']+=bleu1
            self.metrics_gen['bleu2']+=bleu2
            self.metrics_gen['bleu3']+=bleu3
            self.metrics_gen['bleu4']+=bleu4
            self.metrics_gen['count']+=1

        dis1, dis2, dis3, dis4=distinct_metrics(generated)
        self.metrics_gen['dist1']=dis1
        self.metrics_gen['dist2']=dis2
        self.metrics_gen['dist3']=dis3
        self.metrics_gen['dist4']=dis4

    def vector2sentence(self,batch_sen,is_context = False):
        sentences=[]

        if is_context:
            for sen in batch_sen:
                # temp = self.tokenizer.decode(sen[0]).split(" ")
                sentence=[]
                for word_id in sen:
                    # if word_id==self.tokenizer.eos_token_id:
                    #     continue
                    sentence.append(word_id)
                    # elif word==3:
                    #     sentence.append('_UNK_')
                sentence = self.tokenizer.decode(sentence).replace("<|endoftext|>"," ").strip().split(" ")
                sentences.append(sentence)
        else:
            # for sen in batch_sen.numpy().tolist():
            for sen in batch_sen:
                # temp = self.tokenizer.decode(sen[0]).split(" ")
                sentence=[]
                for word_id in sen:
                    if word_id==self.tokenizer.eos_token_id:
                        break
                    sentence.append(word_id)
                    # elif word==3:
                    #     sentence.append('_UNK_')
                sentence = self.tokenizer.decode(sentence).strip().split(" ")
                sentences.append(sentence)
        return sentences

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
    loop=TrainLoop_fusion_gen(vars(args),is_finetune=True)
    loop.train()
    met=loop.val(True)
