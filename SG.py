import pandas as pd
import numpy as np
from itertools import product
import json
from tqdm import tqdm_notebook,tqdm
import torch
import re
import os

class SG:
    def __init__(self, dataset='redial', debug=False):
        self.debug = debug
        self.dataset_dir = os.path.join('sentimentG', dataset)

        # with open(os.path.join(self.dataset_dir, "subRG/DBpediaMovieId2imdbId.json"),'r') as f:
        #     DBpediaMovieId2imdbId = json.load(f)
        with open(os.path.join(self.dataset_dir, "subSG/sub_RG.json"),'r') as f:
            subkg = json.load(f)
        with open(os.path.join(self.dataset_dir, "subSG/SG_2id.json"),'r') as f:
            entity2id = json.load(f)
        with open(os.path.join(self.dataset_dir, "subSG/relation_dict.json"),'r') as f:
            relation2id = json.load(f)
        with open(os.path.join(self.dataset_dir, "subSG/movie2user.json"),'r') as f:
            self.movie2user = json.load(f)
        with open(os.path.join(self.dataset_dir, "subSG/movieUser2entity.json"),'r') as f:
            self.movieUser2entity = json.load(f)
        with open(os.path.join(self.dataset_dir, "subSG/kg2sg.json"),'r') as f:
            self.kg2sg = json.load(f)
        
        self.entity2id = entity2id
        edge_list = set()  # [(entity, entity, relation)]
        # relation = set()
        for entity in tqdm(subkg.keys()):
            # if str(entity) not in self.entity_kg:
            #     continue
            if entity =='null':
                continue
            if entity == 'nan'or str(entity)=='nan':
                continue
            for relation_and_tail in subkg[entity]:
                if relation_and_tail[1] =='null' or str(relation_and_tail[1])=='nan':
                    continue
                # print(entity)
                edge_list.add((entity2id[entity], entity2id[relation_and_tail[1]], relation_and_tail[0]))
                edge_list.add((entity2id[relation_and_tail[1]], entity2id[entity], relation_and_tail[0]))
                # relation.add(relation_and_tail[0])
        edge_list = list(edge_list)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.num_relations = len(relation2id)
        self.pad_entity_id = 0
        self.num_entities = max(entity2id.values()) + 2
        self.kg_item_ids = self.kg2sg['key']
        self.sg_item_ids = self.kg2sg['value']

    def get_entity_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'pad_entity_id': self.pad_entity_id,
            'movie2user':self.movie2user,
            'kg_item_ids':self.kg_item_ids,
            'sg_item_ids': self.sg_item_ids,

        }
        return kg_info
