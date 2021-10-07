from .tag_generic import GeneralEval
import logging
import torch
from torch import nn, optim
from models.VAE.DVAE import VAE,vae_loss
import numpy as np
from torch.utils.data import SubsetRandomSampler,DataLoader
from collections import defaultdict,Counter
from configs.globalConfigs import args as _args
import pickle
import pandas as pd

class VAEEval(GeneralEval):
    def __init__(self,args=_args):
        super().__init__(True,raw = True)
        self.args = args
        self.cuda = args.cuda
        self.cos = nn.CosineSimilarity()
        self.df_idx = self.data_set.df_idx
        self.ext_df =  self.data_set.ext_df
        self.b_df =  self.data_set.bow_df
        self.pred = self.data_set.tag_ids
        self.is_q = 0

        self.init_fe()
        self.init_optim()
        self.init_loss()
        self.init_data_loader()

    def init_fe(self):
        self.feature_extractor = VAE()


    def init_optim(self):
        self.optim = optim.Adam(self.feature_extractor.parameters(),self.args.lr,weight_decay = self.args.weight_decay)

    def init_loss(self):
        self.loss = vae_loss  

    def init_data_loader(self):
        self.ids_loader = DataLoader(self.df_idx,batch_size = self.args.batch_size)
        ext_keys = self.query_ext([k.split() for k in self.data_set.keys])
        keys = [k.split() for k in self.data_set.keys]

        self.ext_keys = [self.data_set.filter_noise(list(Counter(map(int,k)).items())) for k in ext_keys]
        self.raw_keys = [self.data_set.filter_noise(list(Counter(map(int,k)).items())) for k in keys]


    def train(self):
        for i in range(self.args.nepochs): #s
            epoch_loss = 0
            self.feature_extractor.train()
            if not self.is_q:
                for ids in self.ids_loader:
                    text_bow0 = self.ext_df[ids.cpu().numpy()]
                    text_bow1 = self.b_df[ids.cpu().numpy()]
    
                    _pred = self.pred[ids.cpu().numpy()]
                    out_bows,*_,loss,in_bows = self.feature_extractor(text_bow0.tolist(),text_bow1.tolist())
                    
                    epoch_loss += loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
            else:
                out_bows,*_,loss,in_bows = self.feature_extractor(self.ext_keys,self.raw_keys)
                epoch_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()    
            
            if i%10 == 1:
                print(torch.sum(self.cos(out_bows,in_bows)))
            print('epoch:{},loss:{}'.format(i,epoch_loss))
            torch.save(self.feature_extractor,self.feature_extractor.vae_path+str(self.is_q))
