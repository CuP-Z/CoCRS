import pandas as pd
import numpy as np
from itertools import product
import json
from tqdm import tqdm_notebook,tqdm
import torch
import re
import os

class ReviewG:
    def __init__(self, dataset='redial', debug=False):
        self.debug = debug
        self.dataset_dir = os.path.join('RGdata', dataset)

        # with open(os.path.join(self.dataset_dir, "subRG/DBpediaMovieId2imdbId.json"),'r') as f:
        #     DBpediaMovieId2imdbId = json.load(f)
        with open(os.path.join(self.dataset_dir, "subRG/sub_RG.json"),'r') as f:
            movie_subkg = json.load(f)
        with open(os.path.join(self.dataset_dir, "subRG/entity2id.json"),'r') as f:
            entity2id = json.load(f)
        with open(os.path.join(self.dataset_dir, "subRG/sub_relation2ID.json"),'r') as f:
            relation2id = json.load(f)

        edge_list = set()  # [(entity, entity, relation)]
        # relation = set()
        for entity in tqdm(movie_subkg.keys()):
            # if str(entity) not in self.entity_kg:
            #     continue
            if entity =='null':
                continue
            for relation_and_tail in movie_subkg[entity]:
                if relation_and_tail[1] =='null':
                    continue
                edge_list.add((entity2id[entity], entity2id[relation_and_tail[1]], relation_and_tail[0]))
                edge_list.add((entity2id[relation_and_tail[1]], entity2id[entity], relation_and_tail[0]))
                # relation.add(relation_and_tail[0])
        edge_list = list(edge_list)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.num_relations = 2
        self.pad_entity_id = max(entity2id.values()) + 1
        self.num_entities = max(entity2id.values()) + 2
        # self.imdb_ids = DBpediaMovieId2imdbId['value']
        # self.item_ids = DBpediaMovieId2imdbId['key']

    def get_entity_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'pad_entity_id': self.pad_entity_id,
            # 'imdb_ids':self.imdb_ids,
            # 'item_ids': self.item_ids,
        }
        return kg_info
