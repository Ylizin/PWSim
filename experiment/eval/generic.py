import os
from torch import nn, optim
from data_utils.dataloader import WordsDataSet
import logging
import numpy as np

class GeneralEval:
    def __init__(self):
        self.feature_extractor = None
        self.data_set = None
        self.optim = None
        self.init_data_set()

    def init_fe(self):
        pass

    def init_data_set(self):
        self.data_set = WordsDataSet()
        self.data_set.map_word2id()
        # self.data_set.set_train_cols(['sr','rs','rd','ri'])
        self.data_set.set_train_cols(['des'])

        logging.info('{} load data set done.'.format(self.__class__.__name__))
    
    def init_optim(self):
        pass

    def init_data_loader(self):
        pass

    def get_metrics(self,sims_sort,pos_ids,topk):
        sims_sort_np = sims_sort[:topk].numpy()
        pos_ids_np = np.array(pos_ids[:topk])
        inter = np.intersect1d(sims_sort_np,pos_ids_np)
        precision = len(inter)/len(sims_sort_np)
        recall = len(inter)/len(pos_ids_np)
        return precision,recall
    
