from models.transformer import TorchGeneratorModel,_build_encoder,_build_decoder,_build_encoder_mask, _build_encoder4kg, _build_decoder4kg
from models.utils import _create_embeddings,_create_entity_embeddings
from models.graph import SelfAttentionLayer,SelfAttentionLayer_batch
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
import numpy as np
from reviewG import ReviewG
import json
from transformers import AutoTokenizer, AutoModel,BertForPreTraining
from SG import SG
import math
from models.transformer import TransformerEncoder,TransformerDecoder,GPT2forCRS

class GateLayer(nn.Module):
    def __init__(self, input_dim,is_minus=False):
        super(GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)
        self.minus=is_minus

    def forward(self, input1, input2):
        if self.minus:
            norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
            gate = torch.sigmoid(self._norm_layer2(norm_input))*0.2  # (bs, 1)
            gated_emb = input1 - gate * input2
        else:
            norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
            gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
            gated_emb = gate * input1 + (1 - gate) * input2  # (bs, dim)
        return gated_emb

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

EDGE_TYPES = [58, 172]
def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :# and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1

    count_min = 500
    for h, t, r in edge_list:
        if relation_cnt[r] > count_min and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > count_min], len(relation_idx)

def concept_edge_list4GCN():
    # node2index=json.load(open('key2index_3rd.json',encoding='utf-8')) # redial
    # f=open('conceptnet_edges2nd.txt',encoding='utf-8') #redial
    node2index=json.load(open('key2index_inspired.json',encoding='utf-8')) # inspired
    f=open('concept_inspired.txt',encoding='utf-8') #Inspired
    edges=set()
    stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
    for line in f:
        lines=line.strip().split('\t')
        # entity0=node2index[lines[1].split('/')[0]] # Redial
        entity0=node2index[lines[0].split('/')[0]] # Inspired
        entity1=node2index[lines[2].split('/')[0]]
        # if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords: #redial
        if lines[0].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges.add((entity0,entity1))
        edges.add((entity1,entity0))
    edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
    return torch.LongTensor(edge_set).cuda()

class CrossModel(nn.Module):
    def __init__(self, opt, dictionary, tokenizer=None,is_finetune=False, padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        # self.pad_idx = dictionary[dictionary.null_token]
        # self.start_idx = dictionary[dictionary.start_token]
        # self.end_idx = dictionary[dictionary.end_token]
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)

        self.tokenizer = tokenizer

        self.transformer = GPT2forCRS.from_pretrained("/home/zhy/crs/MyCRS-newdata/DialoGPT-small")
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.transformer.config.pad_token_id = self.tokenizer.pad_token_id
        print("GPT over")
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # self.bert = AutoModel.from_pretrained('bert-base-uncased')
        # for p in self.parameters():
        #     p.requires_grad = False
        self.batch_size = opt['batch_size']
        self.max_r_length = opt['max_r_length']

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.concept_embeddings=_create_entity_embeddings(
            opt['n_concept']+1, opt['dim'], 0)
        self.concept_padding=0

        self.kg = pkl.load(
            open("data-{}/subkg.pkl".format(opt['dataset']), "rb")
        )

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder4kg(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )
        self.db_norm = nn.Linear(opt['dim'], opt['embedding_size'])
        self.kg_norm = nn.Linear(opt['dim'], opt['embedding_size'])

        self.db_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.kg_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])

        self.criterion = nn.CrossEntropyLoss(reduce=False)

        self.self_attn = SelfAttentionLayer_batch(opt['dim'], opt['dim'])

        self.self_attn_db = SelfAttentionLayer(opt['dim'], opt['dim'])

        self.user_norm = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm = nn.Linear(opt['dim'], 1)
        self.copy_norm = nn.Linear(opt['embedding_size']*2+opt['embedding_size'], opt['embedding_size'])
        self.representation_bias = nn.Linear(opt['embedding_size'], len(dictionary) + 4)

        self.info_con_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_db_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_output_db = nn.Linear(opt['dim'], opt['n_entity'])
        self.info_output_con = nn.Linear(opt['dim'], opt['n_concept']+1)
        self.info_con_loss = nn.MSELoss(size_average=False,reduce=False)
        self.info_db_loss = nn.MSELoss(size_average=False,reduce=False)

        self.user_representation_to_bias_1 = nn.Linear(opt['dim'], 512)
        self.user_representation_to_bias_2 = nn.Linear(512, len(dictionary) + 4)

        self.entity_classify = nn.Linear(opt['dim'], opt['n_entity'])
        self.output_en = nn.Linear(opt['dim'], opt['n_entity'])

        self.embedding_size=opt['embedding_size']
        self.dim=opt['dim']

        edge_list, self.n_relation = _edge_list(self.kg, opt['n_entity'], hop=2)
        edge_list = list(set(edge_list))
        print(len(edge_list), self.n_relation)
        self.dbpedia_edge_sets=torch.LongTensor(edge_list).cuda()
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        
        self.node_embeds = nn.Parameter(torch.empty(opt['n_entity'], self.dim))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(self.db_edge_idx, requires_grad=False)
        self.edge_type = nn.Parameter(self.db_edge_type, requires_grad=False)


        # self.dbpedia_RGCN=RGCNConv(self.dim, self.dim, self.n_relation, num_bases=opt['num_bases'])
        self.dbpedia_RGCN=RGCNConv(opt['n_entity'], self.dim, self.n_relation, num_bases=opt['num_bases'])
        #self.concept_RGCN=RGCNConv(opt['n_concept']+1, self.dim, self.n_con_relation, num_bases=opt['num_bases'])
        self.concept_edge_sets=concept_edge_list4GCN()
        self.concept_GCN=GCNConv(self.dim, self.dim)
        self.gate_layer1 = GateLayer(self.dim)

        # self.entity_proj1 = nn.Sequential(
        #     nn.Linear(self.dim, self.dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.dim // 2, self.dim),
        # )


        #self.concept_GCN4gen=GCNConv(self.dim, opt['embedding_size'])

        # w2i=json.load(open('word2index_redial2.json',encoding='utf-8'))
        # self.i2w={w2i[word]:word for word in w2i}

        # self.mask4key=torch.Tensor(np.load('mask4key20rev.npy')).cuda()
        # self.mask4movie=torch.Tensor(np.load('mask4movie20rev.npy')).cuda()
        # self.mask4=self.mask4key+self.mask4movie
        if is_finetune:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),
                      self.concept_embeddings.parameters(),
                      self.self_attn.parameters(), self.self_attn_db.parameters(), self.user_norm.parameters(),
                      self.gate_norm.parameters(), self.output_en.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = False

        Rg = ReviewG(dataset=opt['dataset']).get_entity_kg_info()

        RG_edge_index = Rg['edge_index']
        RG_edge_type = Rg['edge_type']
        RG_num_relations= Rg['num_relations']
        RG_num_bases=opt['num_bases']
        RG_n_entity=Rg['num_entities']

        self.RG_encoder = RGCNConv(opt['n_entity'], self.dim, num_relations=RG_num_relations,
                                   num_bases=RG_num_bases)

        self.RG_edge_index = nn.Parameter(RG_edge_index, requires_grad=False)
        self.RG_edge_type = nn.Parameter(RG_edge_type, requires_grad=False)

        self.bert2trans_overview = nn.Linear(768,opt['dim'])
        self.bert2trans = nn.Linear(768,opt['dim'])
        self.bert_conc_norm = nn.Linear(opt['dim']*2, opt['dim'])


        self.self_attn_overview = SelfAttentionLayer(opt['dim'], opt['dim'])

        self.user_over_entity = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm_overview = nn.Linear(opt['dim'], 1)

        sg = SG(opt['dataset']).get_entity_kg_info()
        sg_edge_index = sg['edge_index']
        sg_edge_type = sg['edge_type']
        sg_num_relations= sg['num_relations']
        sg_num_bases=opt['num_bases']
        sg_n_entity=sg['num_entities']
        
        self.sg_encoder = RGCNConv(sg_n_entity, self.dim, num_relations=sg_num_relations,
                                   num_bases=sg_num_bases)
        
        self.sg_edge_index = nn.Parameter(sg_edge_index, requires_grad=False)
        self.sg_edge_type = nn.Parameter(sg_edge_type, requires_grad=False)

        self.self_attn_simi_user = SelfAttentionLayer(opt['dim'], opt['dim'])
        self.simi_user_ = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_simi_user = nn.Linear(opt['dim'], 1)

        self.graph_fusi = nn.Linear(opt['dim']*2, opt['dim'])

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def decode_greedy(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, bsz, maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db, incr_state)
            #batch*1*hidden
            scores = scores[:, -1:, :]
            #scores = self.output(scores)
            kg_attn_norm = self.kg_attn_norm(attention_kg)
            
            db_attn_norm = self.db_attn_norm(attention_db)

            copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), scores], -1))

            # logits = self.output(latent)
            con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
            voc_logits = F.linear(scores, self.embeddings.weight)
            # print(logits.size())
            # print(mem_logits.size())
            #gate = F.sigmoid(self.gen_gate_norm(scores))

            sum_logits = voc_logits + con_logits #* (1 - gate)
            _, preds = sum_logits.max(dim=-1)
            #scores = F.linear(scores, self.embeddings.weight)

            #print(attention_map)
            #print(db_attention_map)
            #print(preds.size())
            #print(con_logits.size())
            #exit()
            #print(con_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])
            #print(voc_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])
            
            #print(torch.topk(voc_logits.squeeze(0).squeeze(0),k=50)[1])

            #sum_logits = scores
            # print(sum_logits.size())

            #_, preds = sum_logits.max(dim=-1)
            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db) #batch*r_l*hidden

        kg_attention_latent=self.kg_attn_norm(attention_kg)

        #map=torch.bmm(latent,torch.transpose(kg_embs_norm,2,1))
        #map_mask=((1-encoder_states_kg[1].float())*(-1e30)).unsqueeze(1)
        #attention_map=F.softmax(map*map_mask,dim=-1)
        #attention_latent=torch.bmm(attention_map,encoder_states_kg[0])

        db_attention_latent=self.db_attn_norm(attention_db)

        #db_map=torch.bmm(latent,torch.transpose(db_embs_norm,2,1))
        #db_map_mask=((1-encoder_states_db[1].float())*(-1e30)).unsqueeze(1)
        #db_attention_map=F.softmax(db_map*db_map_mask,dim=-1)
        #db_attention_latent=torch.bmm(db_attention_map,encoder_states_db[0])

        copy_latent=self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1,seqlen,1), db_attention_latent.unsqueeze(1).repeat(1,seqlen,1), latent],-1))

        #logits = self.output(latent)
        con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
        logits = F.linear(latent, self.embeddings.weight)
        # print(logits.size())
        # print(mem_logits.size())
        #gate=F.sigmoid(self.gen_gate_norm(latent))

        sum_logits = logits+con_logits#*(1-gate)
        _, preds = sum_logits.max(dim=2)
        return logits, preds

    def infomax_loss(self, con_nodes_features, db_nodes_features, con_user_emb, db_user_emb, con_label, db_label, mask):
        #batch*dim
        #node_count*dim
        con_emb=self.info_con_norm(con_user_emb)
        db_emb=self.info_db_norm(db_user_emb)
        con_scores = F.linear(db_emb, con_nodes_features, self.info_output_con.bias)
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias)

        info_db_loss=torch.sum(self.info_db_loss(db_scores,db_label.cuda().float()),dim=-1)*mask.cuda()
        info_con_loss=torch.sum(self.info_con_loss(con_scores,con_label.cuda().float()),dim=-1)*mask.cuda()

        return torch.mean(info_db_loss), torch.mean(info_con_loss)

    def get_RGentity_embeds(self):
        # pre_node_embeds = self.pre_node_embeds
        RG_embeds = self.RG_encoder(None, self.RG_edge_index, self.RG_edge_type)
        # RG_embeds = self.RG_entity_proj(RG_embeds) 
        # self.RG_embeds.data[self.RG_values] = RG_embeds[self.RG_keys]
        # self.RG_embeds.data = RG_embeds
        return RG_embeds

    def get_entity_embeds(self):
        RG_embeds = self.get_RGentity_embeds()
        # entity_embeds = self.dbpedia_RGCN(self.node_embeds, self.edge_index, self.edge_type) + self.node_embeds
        # entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        # entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds + self.RG_entity_proj(RG_embeds) + RG_embeds
        # entity_embeds = self.entity_proj2(entity_embeds)
        
        # entity_embeds = self.graph_fusi(torch.cat([entity_embeds,RG_embeds],dim=-1))
        return entity_embeds + RG_embeds

    def get_bert_word_embed(self,context_raw):
        attention_mask = context_raw==self.tokenizer.pad_token_id
        with torch.no_grad():
            output = self.bert(context_raw,attention_mask=attention_mask)
        embeddings = self.bert2trans(output['pooler_output'])
        return embeddings

    def get_overview_embedding(self,overview):
        attention_mask = overview==self.tokenizer.pad_token_id
        with torch.no_grad():
            output = self.bert(overview,attention_mask=attention_mask)
        embeddings = self.bert2trans_overview(output['pooler_output'])
        return embeddings

    def get_sg_embed(self):
        sg_embeds = self.sg_encoder(None, self.sg_edge_index, self.sg_edge_type)
        # RG_embeds = self.RG_entity_proj(RG_embeds) 
        # self.RG_embeds.data[self.RG_values] = RG_embeds[self.RG_keys]
        # self.RG_embeds.data = RG_embeds
        return sg_embeds

    def compute_barrow_loss(self, view_1_rep, view_2_rep, mu):
        cov_matrix_up = torch.matmul(view_1_rep, view_2_rep.t())
        bs = view_1_rep.shape[0]
        words_down = (view_1_rep * view_1_rep).sum(dim=1).view(bs, 1)
        entities_down = (view_2_rep * view_2_rep).sum(dim=1).view(1, bs)
        words_down = words_down.expand(bs, bs)
        entities_down = entities_down.expand(bs, bs)
        cov_matrix_down = torch.sqrt(words_down * entities_down + 1e-12)
        cov_matrix = cov_matrix_up / cov_matrix_down
        mask_part1 = torch.eye(bs).cuda()
        mask_part2 = torch.ones((bs, bs)).cuda() - mask_part1

        loss_part1 = ((mask_part1 - cov_matrix).diag() * (mask_part1 - cov_matrix).diag()).sum()
        loss_part2 = ((mask_part2 * cov_matrix) * (mask_part2 * cov_matrix)).sum()
        loss = mu * loss_part1 + (1-mu) * loss_part2

        return loss_part1, loss_part2, loss

    def forward(self, input_ids, inputs_mask, ys, mask_ys, concept_mask, db_mask, seed_sets, labels, con_label, db_label, entity_vector, rec,context_raw,response_raw,overview,simi_user_sets,simi_user_vector,conv=False, test=True, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        # if test == False:
        #     # TODO: get rid of longest_label
        #     # keep track of longest label we've ever seen
        #     # we'll never produce longer ones than that during prediction
        #     self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        #xxs = self.embeddings(xs)
        #mask=xs == self.pad_idx
        # encoder_states = prev_enc if prev_enc is not None else self.encoder(xs)

        # graph network
        # db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        db_nodes_features = self.get_entity_embeds()
        con_nodes_features=self.concept_GCN(self.concept_embeddings.weight,self.concept_edge_sets)

        # print(overview)
        # print(overview.shape)
        user_representation_list = []
        # user_overview_list = []
        db_con_mask=[]
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                # user_overview_list.append(torch.zeros(self.dim).cuda())
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]  # torch can reflect
            # print(user_representation.shape)
            user_representation = self.self_attn_db(user_representation)
            # print(user_representation.shape)
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1]))

        # sg_embeddings = self.get_sg_embed()
        # # print(sg_embeddings.shape)
        # simi_user_list = []
        # simi_user_mask = []
        # # print(simi_user_sets)
        # for i, users in enumerate(simi_user_sets):
        #     if users == []:
        #         simi_user_list.append(torch.zeros(self.dim).cuda())
        #         # user_overview_list.append(torch.zeros(self.dim).cuda())
        #         simi_user_mask.append(torch.zeros([1]))
        #         continue
        #     # print(users)
        #     simi_user_representation = sg_embeddings[users]  # torch can reflect
        #     # print(simi_user_representation.shape)
        #     simi_user_representation = self.self_attn_simi_user(simi_user_representation)
        #     simi_user_list.append(simi_user_representation)
        #     simi_user_mask.append(torch.ones([1]))
        #     # 增加overview
        #     # user_overview = []
        #     # for seed in overview:
        #         # print(overview_embed.shape)
        #         # user_overview.append(overview_embed)
        #     # user_overview = torch.stack(user_overview)
        #     # print(overview_embed.shape)
        #     # user_overview = self.self_attn_overview(overview_embed)
        #     # print(user_overview.shape)

        #     # user_overview_list.append(user_overview)
        # # print(simi_user_list)
        # simi_user_emb = torch.stack(simi_user_list)

        db_user_emb=torch.stack(user_representation_list)
        db_con_mask=torch.stack(db_con_mask)
        # user_overview_emb = torch.stack(user_overview_list)

        graph_con_emb=con_nodes_features[concept_mask]
        con_emb_mask=concept_mask==self.concept_padding

        con_user_emb=graph_con_emb
        con_user_emb,attention=self.self_attn(con_user_emb,con_emb_mask.cuda())

        # bert_user_emb = self.get_bert_word_embed(context_raw)

        # con_user_emb = self.bert_conc_norm(torch.cat([con_user_emb,bert_user_emb],dim=-1))


        # user_emb=self.user_norm(torch.cat([con_user_emb,db_user_emb,bert_user_emb],dim=-1))
        # user_emb=self.user_norm(torch.cat([con_user_emb,db_user_emb],dim=-1))
        # uc_gate = F.sigmoid(self.gate_norm(user_emb))

        # overview_embed = self.get_overview_embedding(overview)

        # user_entity_overview = self.user_over_entity(torch.cat([overview_embed,db_user_emb],dim=-1))
        # uv_gate = F.sigmoid(self.gate_norm_overview(user_entity_overview))

        # user_emb = uc_gate * db_user_emb + (1 - uc_gate)/2 * con_user_emb  + (1 - uv_gate)/2 * overview_embed
        # user_emb = uc_gate * db_user_emb + (1 - uc_gate) * con_user_emb  

        # _, info_con_loss, info_db_loss = self.compute_barrow_loss(db_user_emb, con_user_emb, 0.8)
        _, info_con_loss, info_db_loss = self.compute_barrow_loss(db_user_emb, con_user_emb, 0.8)
        # info_con_loss = 0

        user_emb = self.gate_layer1(db_user_emb,con_user_emb)

        # user_similer_uer = self.simi_user_(torch.cat([simi_user_emb,db_user_emb],dim=-1))
        # us_gate = F.sigmoid(self.gate_simi_user(user_similer_uer))

        # user_emb = simi_user_emb*(1- us_gate) + user_emb * us_gate
        # user_emb = simi_user_emb + user_emb
        
        entity_scores = F.linear(user_emb, db_nodes_features, self.output_en.bias)

        # entity_scores = self.entity_classify(user_emb)
        #entity_scores = scores_db * gate + scores_con * (1 - gate)
        #entity_scores=(scores_db+scores_con)/2

        #mask loss
        #m_emb=db_nodes_features[labels.cuda()]
        #mask_mask=concept_mask!=self.concept_padding
        mask_loss=0#self.mask_predict_loss(m_emb, attention, xs, mask_mask.cuda(),rec.float())

        # info_db_loss, info_con_loss=self.infomax_loss(con_nodes_features,db_nodes_features,con_user_emb,db_user_emb,con_label,db_label,db_con_mask)

        #entity_scores = F.softmax(entity_scores.cuda(), dim=-1).cuda()

        rec_loss=self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.cuda())
        #rec_loss=self.klloss(entity_scores.squeeze(1).squeeze(1).float(), labels.float().cuda())
        rec_loss = torch.sum(rec_loss*rec.float().cuda())

        self.user_rep=user_emb

        #generation---------------------------------------------------------------------------------------------------
        # con_nodes_features4gen=con_nodes_features#self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
        # con_emb4gen = con_nodes_features4gen[concept_mask]
        # con_mask4gen = concept_mask != self.concept_padding
        # #kg_encoding=self.kg_encoder(con_emb4gen.cuda(),con_mask4gen.cuda())
        # kg_encoding=(self.kg_norm(con_emb4gen),con_mask4gen.cuda())

        # db_emb4gen=db_nodes_features[entity_vector] #batch*50*dim
        # db_mask4gen=entity_vector!=0
        # #db_encoding=self.db_encoder(db_emb4gen.cuda(),db_mask4gen.cuda())
        # db_encoding=(self.db_norm(db_emb4gen),db_mask4gen.cuda())

        # if test == False:
        #     # use teacher forcing
        #     scores, preds = self.decode_forced(encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb, mask_ys)
        #     gen_loss = torch.mean(self.compute_loss(scores, mask_ys))

        # else:
        #     scores, preds = self.decode_greedy(
        #         encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb,
        #         bsz,
        #         maxlen or self.longest_label
        #     )
        #     gen_loss = None
        bsz = len(input_ids)
        preds = None
        conv_loss = None
        input_dic = {
            "input_ids":input_ids,
            "attention_mask":inputs_mask,
        }
        scores,preds,conv_loss = None,None,None
        # print(input_ids.shape)
        # print(inputs_mask.shape)
        # print(ys.shape)

        if conv:
            results = self.transformer(**input_dic, conv=True,
                         conv_labels=ys)
            conv_loss,preds = results.conv_loss,results.logits
        
        return scores, preds, entity_scores, rec_loss, conv_loss, mask_loss, info_db_loss, info_con_loss

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        # no support for incremental decoding at this time
        return None

    def compute_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.criterion(output_view.cuda(), score_view.cuda())
        return loss

    def save_model(self):
        torch.save(self.state_dict(), 'saved_model/net_parameter1.pkl')

    def load_model(self):
        self.load_state_dict(torch.load('saved_model/net_parameter1.pkl'))

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        up_bias = self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_rep)))
        # up_bias = self.user_representation_to_bias_3(F.relu(self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_representation)))))
        # Expand to the whole sequence
        up_bias = up_bias.unsqueeze(dim=1)
        output += up_bias
        return output
