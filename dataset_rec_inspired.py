import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
from transformers import AutoTokenizer
from SG import SG
from tqdm import tqdm


class dataset(object):
    def __init__(self,filename,opt,mode='train',pre_train=False,max_turn=9):
        self.max_turn = max_turn
        self.train = False
        if "train" in filename:
            self.train = True
        self.pre_train = pre_train
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.data_type = opt['dataset']
        self.sg = SG(dataset=opt['dataset'])
        self.entity2entityId=pkl.load(open('data-{}/entity2entityId.pkl'.format(opt['dataset']),'rb'))
        self.entityId2entity = {}
        for k,v in self.entity2entityId.items():
            self.entityId2entity[v] = k
        self.entity_max=len(self.entity2entityId)
        

        self.id2entity=pkl.load(open('data-{}/id2entity.pkl'.format(opt['dataset']),'rb'))
        self.subkg=pkl.load(open('data-{}/subkg.pkl'.format(opt['dataset']),'rb'))    #need not back process
        
        if self.data_type == "redial":
            self.text_dict=pkl.load(open('data-{}/text_dict_new.pkl'.format(opt['dataset']),'rb'))
        # self.text_dict=pkl.load(open('data-{}/text_dict_new.pkl'.format(opt['dataset']),'rb'))

        self.movie_set = set()
        for v in self.id2entity.values():
            if v== None:
                continue
            self.movie_set.add(self.entity2entityId[v])

        # self.movie2overview = json.load(open('./movie_overview.json','r'))
        self.movie2overview = {}
        self.batch_size=opt['batch_size']
        self.max_c_length=opt['max_c_length']
        self.max_r_length=opt['max_r_length']
        self.max_count=opt['max_count']
        self.entity_num=opt['n_entity']
        #self.word2index=json.load(open('word2index.json',encoding='utf-8'))
        self.max_review_entity = 5
        f=open(filename,encoding='utf-8')
        self.data=[]
        self.corpus=[]
        if opt['dataset'] == 'redial':
            for line in tqdm(f):
                lines=json.loads(line.strip())
                seekerid=lines["initiatorWorkerId"]
                recommenderid=lines["respondentWorkerId"]
                contexts=lines['messages']
                movies=lines['movieMentions']
                altitude=lines['respondentQuestions']
                initial_altitude=lines['initiatorQuestions']
                cases=self._context_reformulate(contexts,movies,altitude,initial_altitude,seekerid,recommenderid)
                self.data.extend(cases)
        else:
            for line in tqdm(f):
                lines=json.loads(line.strip())
                # print(lines)
                seekerid='SEEKER'
                recommenderid="RECOMMENDER"
                contexts=lines['messages']
                movies=lines['movieMentions']
                altitude=lines['movie_label']
                initial_altitude=None
                cases=self._context_reformulate(contexts,movies,altitude,initial_altitude,seekerid,recommenderid)
                self.data.extend(cases)
        #if 'train' in filename:

        #self.prepare_word2vec()
        self.word2index = json.load(open('word2index.json', encoding='utf-8'))
        # self.key2index=json.load(open('key2index_3rd.json',encoding='utf-8'))
        self.key2index=json.load(open('key2index_inspired.json',encoding='utf-8'))
        

        self.stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])

        #self.co_occurance_ext(self.data)
        #exit()

    def prepare_word2vec(self):
        import gensim
        model=gensim.models.word2vec.Word2Vec(self.corpus,size=300,min_count=1)
        model.save('word2vec_redial')
        word2index = {word: i + 4 for i, word in enumerate(model.wv.index2word)}
        #word2index['_split_']=len(word2index)+4
        #json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
        word2embedding = [[0] * 300] * 4 + [model[word] for word in word2index]+[[0]*300]
        import numpy as np
        
        word2index['_split_']=len(word2index)+4
        json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)

        # print(np.shape(word2embedding))
        np.save('word2vec_redial.npy', word2embedding)

    def padding_w2v(self,sentence,max_length,transformer=True,pad=0,end=2,unk=3):
        vector=[]
        concept_mask=[]
        dbpedia_mask=[]
        for word in sentence:
            vector.append(self.word2index.get(word,unk))
            #if word.lower() not in self.stopwords:
            concept_mask.append(self.key2index.get(word.lower(),0))
            #else:
            #    concept_mask.append(0)
            if '@' in word:
                try:
                    entity = self.id2entity[int(word[1:])]
                    id=self.entity2entityId[entity]
                except:
                    id=self.entity_max
                dbpedia_mask.append(id)
            else:
                dbpedia_mask.append(self.entity_max)
        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)

        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length,concept_mask[-max_length:],dbpedia_mask[-max_length:]
            else:
                return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length]
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],dbpedia_mask+(max_length-len(vector))*[self.entity_max]

    def padding_context(self,contexts,pad=0,transformer=True):
        vectors=[]
        vec_lengths=[]
        if transformer==False:
            if len(contexts)>self.max_count:
                for sen in contexts[-self.max_count:]:
                    vec,v_l=self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors,vec_lengths,self.max_count
            else:
                length=len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors+(self.max_count-length)*[[pad]*self.max_c_length],vec_lengths+[0]*(self.max_count-length),length
        else:
            contexts_com=[]
            for sen in contexts[-self.max_count:-1]:
                contexts_com.extend(sen)
                contexts_com.append('_split_')
            contexts_com.extend(contexts[-1])
            vec,v_l,concept_mask,dbpedia_mask=self.padding_w2v(contexts_com,self.max_c_length,transformer)
            return vec,v_l,concept_mask,dbpedia_mask,0


    def padding_w2v_raw(self,sentence,max_length,transformer=True,pad=0):
        vector=[]
        # print(sentence)
        vector = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(sentence))

        vector.append(self.bert_tokenizer.cls_token_id)

        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length
            else:
                return vector[:max_length],max_length
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length


    def padding_context_raw(self,contexts,pad=0,transformer=True):
        vectors=[]
        vec_lengths=[]
        contexts_com=[]
        sentence = ""
        # for sen in contexts[-self.max_count:-1]:
        #     for word in sen:
        #         sentence+= word+" "
        #     sentence += "[SEP] "
        for word in contexts[-1][:-1]:
            sentence+= word+" "
        sentence += contexts[-1][-1]



        vec,v_l=self.padding_w2v_raw(sentence,self.max_c_length,transformer)
        return vec,v_l


    def response_delibration(self,response,unk='MASKED_WORD'):
        new_response=[]
        for word in response:
            if word in self.key2index:
                new_response.append(unk)
            else:
                new_response.append(word)
        return new_response

    def data_process(self,is_finetune=False):
        data_set = []
        context_before = []
        for line in tqdm(self.data):
            #if len(line['contexts'])>2:
            #    continue
            if is_finetune and line['contexts'] == context_before:
                continue
            else:
                context_before = line['contexts']
            context,c_lengths,concept_mask,dbpedia_mask,_=self.padding_context(line['contexts'])
            context_raw,c_raw_length = self.padding_context_raw(line['contexts_raw'])
            response,r_length,_,_=self.padding_w2v(line['response'],self.max_r_length)
            response_raw,r_raw_length = self.padding_w2v_raw(line['response_raw'],self.max_r_length)

            overview,o_length = self.padding_w2v_raw(line['overview'],self.max_c_length)
            if False:
                mask_response,mask_r_length,_,_=self.padding_w2v(self.response_delibration(line['response']),self.max_r_length)
            else:
                mask_response, mask_r_length=response,r_length
            assert len(context)==self.max_c_length
            assert len(concept_mask)==self.max_c_length
            assert len(dbpedia_mask)==self.max_c_length

            data_set.append([np.array(context),c_lengths,np.array(response),r_length,np.array(mask_response),mask_r_length,line['entity'],
                             line['movie'],concept_mask,dbpedia_mask,line['rec'],
                             np.array(context_raw),c_raw_length,np.array(response_raw),r_raw_length,np.array(overview),o_length,line['simi_user'],line['simi_user_entity'],line['cur_entity']])
        return data_set

    def co_occurance_ext(self,data):
        stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
        keyword_sets=set(self.key2index.keys())-stopwords
        movie_wordset=set()
        for line in data:
            movie_words=[]
            if line['rec']==1:
                for word in line['response']:
                    if '@' in word:
                        try:
                            num=self.entity2entityId[self.id2entity[int(word[1:])]]
                            movie_words.append(word)
                            movie_wordset.add(word)
                        except:
                            pass
            line['movie_words']=movie_words
        new_edges=set()
        for line in data:
            if len(line['movie_words'])>0:
                before_set=set()
                after_set=set()
                co_set=set()
                for sen in line['contexts']:
                    for word in sen:
                        if word in keyword_sets:
                            before_set.add(word)
                        if word in movie_wordset:
                            after_set.add(word)
                for word in line['response']:
                    if word in keyword_sets:
                        co_set.add(word)

                for movie in line['movie_words']:
                    for word in list(before_set):
                        new_edges.add('co_before'+'\t'+movie+'\t'+word+'\n')
                    for word in list(co_set):
                        new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in line['movie_words']:
                        if word!=movie:
                            new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in list(after_set):
                        new_edges.add('co_after'+'\t'+word+'\t'+movie+'\n')
                        for word_a in list(co_set):
                            new_edges.add('co_after'+'\t'+word+'\t'+word_a+'\n')
        f=open('co_occurance.txt','w',encoding='utf-8')
        f.writelines(list(new_edges))
        f.close()
        json.dump(list(movie_wordset),open('movie_word.json','w',encoding='utf-8'),ensure_ascii=False)
        print(len(new_edges))
        print(len(movie_wordset))

    def entities2ids(self,entities):
        return [self.entity2entityId[word] for word in entities]

    def detect_movie(self,sentence,movies):
        raw_text = ""
        token_text = word_tokenize(sentence)
        num=0
        token_text_com=[]
        while num<len(token_text):
            if token_text[num]=='@' and num+1<len(token_text):
                token_text_com.append(token_text[num]+token_text[num+1])
                try:
                    raw_text += " "+ movies[token_text[num+1]]
                except:
                    raw_text += " "+token_text[num+1]
                num+=2
            else:
                token_text_com.append(token_text[num])
                raw_text += " " +token_text[num]
                num+=1
        movie_rec = []
        for word in token_text_com:
            if word[1:] in movies:
                movie_rec.append(word[1:])
        movie_rec_trans=[]
        for movie in movie_rec:
            entity = self.id2entity[int(movie)]
            try:
                movie_rec_trans.append(self.entity2entityId[entity])
            except:
                pass
        return token_text_com,movie_rec_trans,raw_text

    def _context_reformulate(self,context,movies,altitude,ini_altitude,s_id,re_id):
        ##
        # seek_like = {}
        # if altitude == []:
        #     altitude = {}
        # for key in altitude.keys():
        #     if altitude[key]['liked']==0:
        #         seek_like[self.id2entity[int(key)]] = 0
        #     elif altitude[key]['liked']==1:
        #         seek_like[self.id2entity[int(key)]] = 1
        #     else:
        #         if altitude[key]['seen']==1:
        #             seek_like[self.id2entity[int(key)]] = 1
        #         else:
        #             if altitude[key]['suggested']==1:
        #                 seek_like[self.id2entity[int(key)]]=1
        #             else:
        #                 seek_like[self.id2entity[int(key)]]=0
        seek_like = {}
        if self.data_type =='redial':
            if altitude == []:
                altitude = {}
            for key in altitude.keys():
                if altitude[key]['liked']==0:
                    seek_like[self.id2entity[int(key)]] = 0
                elif altitude[key]['liked']==1:
                    seek_like[self.id2entity[int(key)]] = 1
                else:
                    if altitude[key]['seen']==1:
                        seek_like[self.id2entity[int(key)]] = 1
                    else:
                        if altitude[key]['suggested']==1:
                            seek_like[self.id2entity[int(key)]]=1
                        else:
                            seek_like[self.id2entity[int(key)]]=0
        else:
            for key in altitude.keys():
                if altitude[key] == "accept_rating_good" or altitude[key] == "accept_rating_mod" or altitude[key]=="accept_others":
                    seek_like[key] = 1
                else:
                    seek_like[key] = 0
        last_id=None
        #perserve the list of dialogue
        context_list=[]
        for message in context:
            if self.data_type == "redial":
                entities=[]
                for entity in message['entity']:
                    try:
                        entities.append(self.entity2entityId[entity])
                    except:
                        pass
                
                # try:
                #     for entity in self.text_dict[message['text']]:
                #         try:
                #             entities.append(self.entity2entityId[entity])
                #         except:
                #             pass
                # except:
                #     pass
                token_text,movie_rec,raw_text=self.detect_movie(message['text'],movies)
                movie_rec = []
                for moive in message['movie']:
                    try:
                        movie_rec.append(self.entity2entityId[moive])
                    except:
                        pass
            else:
                entities=[]
                for entity in message['entity_link']:
                    try:
                        entities.append(self.entity2entityId[entity])
                    except:
                        pass
                token_text = message['text']
                movie_rec = []
                for moive in message['movie_link']:
                    movie_rec.append(self.entity2entityId[moive])
                for movie_name in message['movie_name']:
                    token_text = token_text.replace(movie_name,'<movie>')
                raw_text = message['text']
            if self.data_type=='inspired':
                message['senderWorkerId'] = message['role']
                token_text = word_tokenize(token_text)
            if len(context_list)==0:
                context_dict={'text':token_text,'entity':entities+movie_rec,'user':message['senderWorkerId'],'movie':movie_rec,'text_raw':raw_text}
                # context_dict={'text':token_text,'entity':entities,'user':message['senderWorkerId'],'movie':movie_rec,'text_raw':raw_text}
                context_list.append(context_dict)
                last_id=message['senderWorkerId']
                continue
            if message['senderWorkerId']==last_id:
                context_list[-1]['text_raw'] +=  raw_text
                context_list[-1]['text']+=token_text
                context_list[-1]['entity']+=entities+movie_rec
                # context_list[-1]['entity']+=entities
                context_list[-1]['movie']+=movie_rec
            else:
                context_dict = {'text': token_text, 'entity': entities+movie_rec,'user': message['senderWorkerId'], 'movie':movie_rec,'text_raw':raw_text}
                # context_dict = {'text': token_text, 'entity': entities,'user': message['senderWorkerId'], 'movie':movie_rec,'text_raw':raw_text}
                context_list.append(context_dict)
                last_id = message['senderWorkerId']
                
                # ##  冷启动
                # if len(movie_rec) != 0 and message['senderWorkerId'] == re_id:
                #     break

        cases=[]
        contexts=[]
        contexts_raw=[]
        entities_set=set()
        entities=[]
        movie_set_pre = set()
        overview = ""
        simi_user = []
        cur_entity = []
        simi_user_entitys = []
        turn_entity_nums = []
        turn_id_textid = [0]
        turn_id = 0
        # self.max_turn = 9 
        id2cur_entity_num = []
        id2_entity_num = []
        last_cur_num = 0
        last_entity_num = 0
        for id,context_dict in enumerate(context_list):
            self.corpus.append(context_dict['text'])
            
            if context_dict['user']==re_id and len(contexts)>0:
                last_cur_num = len(cur_entity)
                response=context_dict['text']
                response_raw = context_dict['text_raw']

                # if len(context_dict['entity']) !=0:
                #     for word in context_dict['entity']:

                # if len(context_dict['entity']) !=0:
                #     for word in context_dict['entity']:
                #         if word in self.movie_set and word not in movie_set_pre:
                #             movie_set_pre.add(word)
                #             movie_entity = self.entityId2entity[word]
                #             if movie_entity == None :
                #                 continue
                #             if movie_entity in self.movie2overview.keys():
                #                 # print(self.movie2overview[movie_entity])
                #                 if str(self.movie2overview[movie_entity]) == 'nan':
                #                     overview += ""
                #                 else:
                #                     overview += "[SEP]" + self.movie2overview[movie_entity]
                #             # print(movie_entity)
                #             if movie_entity in self.sg.movie2user.keys():
                #                 temp_simi_user = self.sg.movie2user[movie_entity]
                #             else:
                #                 temp_simi_user = [[],[]]
                #             # print(temp_simi_user)
                #             # print(overview)
                #             if movie_entity in seek_like.keys():
                #                 senti = seek_like[movie_entity]
                #                 # print(senti)
                #                 for user in temp_simi_user[senti]:
                #                     simi_user.append(self.sg.entity2id[user])

                if len(context_dict['movie'])!=0:
                    # if self.train:
                    #     cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': movie, 'rec':1,'contexts_raw':deepcopy(contexts_raw),'response_raw':deepcopy(response_raw),'overview':overview,'simi_user':deepcopy(simi_user)})
                    # else:
                    # if self.pre_train:
                    #     label = context_dict['movie'] + context_dict['entity']
                    # else:
                    
                    turn_id +=1
                    turn_entity_nums.append(len(cur_entity))
                    turn_id_textid.append(id)


                    label = context_dict['movie']

                    for movie in label:
                        # print('simi_user_entitys',simi_user_entitys)
                        # print('cur_entity',cur_entity)
                        # print('movie',movie)
                        if movie in simi_user_entitys:
                            simi_user_entitys.remove(movie)
                        if movie in cur_entity:
                            cur_entity.remove(movie)
                        #if movie not in entities_set:
                        # cases.append({'contexts': deepcopy(contexts[-(self.max_turn*2+1):]), 'response': response, 'entity': deepcopy(entities[-sum(id2_entity_num[-(self.max_turn*2+1):]):]), 'movie': movie, 'rec':1,'contexts_raw':deepcopy(contexts_raw),'response_raw':deepcopy(response_raw),'overview':overview,'simi_user':deepcopy(simi_user),'simi_user_entity':simi_user_entitys,'cur_entity':cur_entity[-sum(id2cur_entity_num[-(self.max_turn*2+1):]):]})
                        cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': movie, 'rec':1,'contexts_raw':deepcopy(contexts_raw),'response_raw':deepcopy(response_raw),'overview':overview,'simi_user':deepcopy(simi_user),'simi_user_entity':simi_user_entitys,'cur_entity':cur_entity})
                    # entities.extend(cur_entity)
                    cur_entity = []
                    last_cur_num = 0
                else:
                    # if self.pre_train:
                    #     label = context_dict['movie'] + context_dict['entity']
                    #     for movie in label:
                    #         cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': movie, 'rec':1,'contexts_raw':deepcopy(contexts_raw),'response_raw':deepcopy(response_raw),'overview':overview,'simi_user':deepcopy(simi_user),'simi_user_entity':simi_user_entitys,'cur_entity':cur_entity})
                    # else:
                    # cases.append({'contexts': deepcopy(contexts[-(self.max_turn*2+1):]), 'response': response, 'entity': deepcopy(entities[-sum(id2_entity_num[-(self.max_turn*2+1):]):]), 'movie': 0, 'rec':0,'contexts_raw':deepcopy(contexts_raw),'response_raw':deepcopy(response_raw),'overview':overview,'simi_user':deepcopy(simi_user),'simi_user_entity':simi_user_entitys,'cur_entity':cur_entity[-sum(id2cur_entity_num[-(self.max_turn*2+1):]):]})
                    cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': 0, 'rec':0,'contexts_raw':deepcopy(contexts_raw),'response_raw':deepcopy(response_raw),'overview':overview,'simi_user':deepcopy(simi_user),'simi_user_entity':simi_user_entitys,'cur_entity':cur_entity})
                    # pass
                # cur_entity = []
                turn_id +=1
                contexts_raw.append(context_dict['text_raw'])
                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    # if word not in entities_set:
                    entities.append(word)
                    cur_entity.append(word)
                    #     entities_set.add(word)

                # max_entity_num = 2
                # entities = entities[-max_entity_num:]
                # cur_entity = cur_entity[-max_entity_num:]

                cur_simi_user_set = set()
                # for movie in entities:

                for movie in context_dict['movie']:
                    # entities.append(movie)
                    # cur_entity.append(movie)
                # for word in context_dict['entity']:
                    if movie in self.movie_set and movie not in movie_set_pre:
                        movie_set_pre.add(movie)
                        movie_entity = self.entityId2entity[movie]
                        if movie_entity == None :
                            continue
                        if movie_entity in self.movie2overview.keys():
                            # print(self.movie2overview[movie_entity])
                            if str(self.movie2overview[movie_entity]) == 'nan':
                                overview += ""
                            else:
                                overview += "[SEP]" + self.movie2overview[movie_entity]
                        # print(movie_entity)
                        if movie_entity in self.sg.movie2user.keys():
                            temp_simi_user = self.sg.movie2user[movie_entity]
                        else:
                            temp_simi_user = [[],[]]
                        # print(temp_simi_user)
                        # print(overview)
                        if movie_entity in seek_like.keys():
                            senti = seek_like[movie_entity]
                            # print(senti)
                            for user in temp_simi_user[senti]:
                                cur_simi_user_set.add(self.sg.entity2id[user])
                                # simi_user.append(self.sg.entity2id[user])
                                if user in self.sg.movieUser2entity[movie_entity][senti].keys():
                                    for entity in self.sg.movieUser2entity[movie_entity][senti][user][:self.max_review_entity]:
                                        entity = self.entity2entityId[entity]
                                        if (entity in self.movie_set):
                                            # movie
                                            if entity not in cur_entity:
                                                cur_entity.append(entity)
                                        if entity not in entities_set:
                                            simi_user_entitys.append(entity)
                                            # cur_entity.append(entity)
                                            # entities.append(entity)
                                            entities_set.add(entity)
                if len(seek_like) == 0:
                    simi_user = list(cur_simi_user_set)
                else:
                    simi_user = list(set(simi_user) | cur_simi_user_set)
                        # for user in simi_user:
                        #     entities.extend(self.sg.movieUser2entity[movie][senti][user])
                    # entities_set.add(word)
                    # if word in self.movie_set and word not in movie_set_pre:
                    #     movie_set_pre.add(word)
                    #     movie_entity = self.entityId2entity[word]
                    #     if movie_entity == None :
                    #         continue
                    #     if movie_entity in self.movie2overview.keys():
                    #         # print(self.movie2overview[movie_entity])
                    #         if str(self.movie2overview[movie_entity]) == 'nan':
                    #             overview += ""
                    #         else:
                    #             overview += "[SEP]" + self.movie2overview[movie_entity]
                    #     # print(movie_entity)
                    #     if movie_entity in self.sg.movie2user.keys():
                    #         temp_simi_user = self.sg.movie2user[movie_entity]
                    #     else:
                    #         temp_simi_user = [[],[]]
                    #     # print(temp_simi_user)
                    #     # print(overview)
                    #     if movie_entity in seek_like.keys():
                    #         senti = seek_like[movie_entity]
                    #         # print(senti)
                    #         for user in temp_simi_user[senti]:
                    #             simi_user.append(self.sg.entity2id[user])
                    # else:
                    #     overview += ""
                # temp_num = len(cur_entity)-last_cur_num
                # if temp_num>0:
                id2cur_entity_num.append(len(cur_entity)-last_cur_num)
                # else:
                #     id2cur_entity_num.append(len(cur_entity))

            else:
                # cur_entity = []
                last_cur_num = len(cur_entity)

                contexts_raw.append(context_dict['text_raw'])
                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    # if word not in entities_set:
                    cur_entity.append(word)
                    entities.append(word)
                    #     entities_set.add(word)
                cur_simi_user_set = set()
                for movie in context_dict['movie']:
                    # entities.append(movie)
                    # cur_entity.append(movie)
                # for word in context_dict['entity']:
                    if movie in self.movie_set and movie not in movie_set_pre:
                        movie_set_pre.add(movie)
                        movie_entity = self.entityId2entity[movie]
                        if movie_entity == None :
                            continue
                        if movie_entity in self.movie2overview.keys():
                            # print(self.movie2overview[movie_entity])
                            if str(self.movie2overview[movie_entity]) == 'nan':
                                overview += ""
                            else:
                                overview +=  self.movie2overview[movie_entity] + "[SEP]" 
                        # print(movie_entity)
                        if movie_entity in self.sg.movie2user.keys():
                            temp_simi_user = self.sg.movie2user[movie_entity]
                        else:
                            temp_simi_user = [[],[]]
                        # print(temp_simi_user)
                        # print(overview)
                        if movie_entity in seek_like.keys():
                            senti = seek_like[movie_entity]
                            # print(senti)
                            for user in temp_simi_user[senti]:
                                # simi_user.append(self.sg.entity2id[user])
                                # print(senti)
                                # print(movie_entity)
                                cur_simi_user_set.add(self.sg.entity2id[user])
                                # print(user)
                                if user in self.sg.movieUser2entity[movie_entity][senti].keys():
                                    for entity in self.sg.movieUser2entity[movie_entity][senti][user][:self.max_review_entity]:
                                        entity = self.entity2entityId[entity]
                                        if (entity in self.movie_set):
                                            if entity not in cur_entity:
                                                cur_entity.append(entity)
                                        if entity not in entities_set:
                                            simi_user_entitys.append(entity)
                                            # cur_entity.append(entity)
                                            # entities.append(entity)
                                            entities_set.add(entity)
                if len(seek_like) == 0:
                    simi_user = list(cur_simi_user_set)
                else:
                    simi_user = list(set(simi_user) | cur_simi_user_set)
                        # for user in simi_user:
                        #     entities.extend(self.sg.movieUser2entity[movie][senti][user])
                    # if word in self.movie_set and word not in movie_set_pre:
                    #     movie_set_pre.add(word)
                    #     movie_entity = self.entityId2entity[word]
                    #     if movie_entity == None :
                    #         continue
                    #     if movie_entity in self.movie2overview.keys():
                    #         # print(self.movie2overview[movie_entity])
                    #         if str(self.movie2overview[movie_entity]) == 'nan':
                    #             overview += ""
                    #         else:
                    #             overview += "[SEP]" + self.movie2overview[movie_entity]
                    #     # print(movie_entity)
                    #     if movie_entity in self.sg.movie2user.keys():
                    #         temp_simi_user = self.sg.movie2user[movie_entity]
                    #     else:
                    #         temp_simi_user = [[],[]]
                    #     # print(temp_simi_user)
                    #     # print(overview)
                    #     if movie_entity in seek_like.keys():
                    #         senti = seek_like[movie_entity]
                    #         # print(senti)
                    #         for user in temp_simi_user[senti]:
                    #             simi_user.append(self.sg.entity2id[user])
                    # if word in self.movie_set:
                    #     movie_entity = self.entityId2entity[word]
                    #     if movie_entity == None :
                    #         continue
                    #     if movie_entity in self.movie2overview.keys():
                    #         # print(self.movie2overview[movie_entity])
                    #         if str(self.movie2overview[movie_entity]) == 'nan':
                    #             overview += ""
                    #         else:
                    #             overview += "[SEP]" + self.movie2overview[movie_entity]
                    #     # print(movie_entity)
                    #     if movie_entity in self.sg.movie2user.keys():
                    #         temp_simi_user = self.sg.movie2user[movie_entity]
                    #     else:
                    #         temp_simi_user = [[],[]]
                    #     # print(temp_simi_user)
                    #     # print(overview)
                    #     if movie_entity in seek_like.keys():
                    #         senti = seek_like[movie_entity]
                    #         # print(senti)
                    #         for user in temp_simi_user[senti]:
                    #             simi_user.append(self.sg.entity2id[user])
                    # else:
                    #     overview += ""
            id2cur_entity_num.append(len(cur_entity)-last_cur_num)
            # turn_id += 1
            id2_entity_num.append(len(context_dict['entity']))    
        return cases
# sg = SG(dataset='inspired')
# sg = SG(dataset='redial')
class CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num):
        self.data=dataset
        self.entity_num = entity_num
        self.concept_num = concept_num+1
        self.max_simi_user_num = 200
        self.max_num_entities = 200
        # self.sg = SG(dataset='redial')
        self.sg = SG(dataset='inspired')

    def __getitem__(self, index):
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        context, c_lengths, response, r_length, mask_response, mask_r_length, entity, movie, concept_mask, dbpedia_mask, rec, context_raw,c_raw_length,response_raw,r_raw_length,overview,o_length,simi_user,simi_user_entity,cur_entity = self.data[index]
        entity_vec = np.zeros(self.entity_num)
        simi_user_entity_vec = np.zeros(self.entity_num)
        cur_entity_vec = np.zeros(self.entity_num)
        entity_vector=np.zeros(200,dtype=np.int32)
        simi_user_vec = np.zeros(self.sg.num_entities)
        simi_user_vector=np.zeros(self.max_simi_user_num,dtype=np.int32)
        
        point=0
        for en in entity[-200:]:
            entity_vec[en]=1
            entity_vector[point]=en
            point+=1
        
        point=0
        for en in simi_user_entity[-1:]:
            simi_user_entity_vec[en]=1
            point+=1
        point=0
        for en in cur_entity[-200:]:
            cur_entity_vec[en]=1
            point+=1

        point=0
        for en in simi_user[-self.max_simi_user_num:]:
            simi_user_vec[en]=1
            simi_user_vector[point]=en
            point+=1
            # if point>self.max_simi_user_num:
            #     break

        concept_vec=np.zeros(self.concept_num)
        for con in concept_mask:
            if con!=0:
                concept_vec[con]=1

        db_vec=np.zeros(self.entity_num)
        for db in dbpedia_mask:
            if db!=0:
                db_vec[db]=1

        return context, c_lengths, response, r_length, mask_response, mask_r_length, entity_vec, entity_vector, movie, np.array(concept_mask), np.array(dbpedia_mask), concept_vec, db_vec, rec,context_raw,c_raw_length,response_raw,r_raw_length,overview,o_length,simi_user_vec,simi_user_vector,simi_user_entity_vec,cur_entity_vec

    def __len__(self):
        return len(self.data)

# if __name__=='__main__':
#     ds=dataset('data-redial/train_data.jsonl')
#     print()
