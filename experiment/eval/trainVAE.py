from .tag_generic import GeneralEval
import logging
import torch
from torch import nn, optim
from models.VAE.VAE import VAE,vae_loss
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
        self.df =  self.data_set.ext_df
#         self.df =  self.data_set.bow_df
        # self.pred = self.data_set.main_ids
        
        self.is_serv = 1
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
        self.keys = [self.data_set.filter_noise(list(Counter(map(int,k)).items())) for k in ext_keys]

    def train(self):
        for i in range(self.args.nepochs): #s
            epoch_loss = 0
            self.feature_extractor.train()
            if self.is_serv:
                for ids in self.ids_loader:
                    text_bow = self.df[ids.cpu().numpy()]
                    # _pred = self.pred[ids.cpu().numpy()]
                    out_bows,*_,mu,log_var,in_bows = self.feature_extractor(text_bow.tolist())
                    loss = self.loss(in_bows,out_bows,mu,log_var)   
                    
                    epoch_loss += loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
            # else: # train them together 
                out_bows,*_,mu,log_var,in_bows = self.feature_extractor(self.keys)
                loss = self.loss(in_bows,out_bows,mu,log_var)   
                epoch_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()    
            
            if i%10 == 1:
                print(torch.sum(self.cos(out_bows,in_bows)))
            print('epoch:{},loss:{}'.format(i,epoch_loss))
            torch.save(self.feature_extractor,self.feature_extractor.vae_path)# +str(self.is_serv))
