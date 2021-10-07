from .tag_generic import GeneralEval
import logging
import torch
from torch import nn, optim
from models.Atten.MVLSTM import MVLSTM
import numpy as np
from torch.utils.data import SubsetRandomSampler,DataLoader
from collections import defaultdict
from configs.globalConfigs import args as _args
import pickle
import pandas as pd

class LSTMEval(GeneralEval):
    def __init__(self,args=_args,train_test_id=0):
        super().__init__(train_test_id=train_test_id)
        self.args = args
        self.cuda = args.cuda
        self.cos = nn.CosineSimilarity()
        self.train_pairs = self.data_set.train_pairs    
        self.test_keys = self.data_set.test_keys
        self.df_idx = self.data_set.df_idx
        self.model_str = "TagMVLSTMEval"

        self.init_fe()
        self.init_optim()
        self.init_loss()
        self.init_data_loader()

    def init_fe(self):
        self.feature_extractor = MVLSTM()

    def init_optim(self):
        self.optim = optim.Adam(self.feature_extractor.parameters(),self.args.lr,weight_decay = self.args.weight_decay)

    def init_loss(self):
        self.loss = nn.MSELoss()    

    def init_data_loader(self):
        def collate_f(batch):
            id1 = [b[0] for b in batch]
            id2 = [b[1] for b in batch]
            l = [b[2] for b in batch]
            l = torch.LongTensor(l)
            return id1,id2,l
        self.train_data_loader = DataLoader(self.train_pairs,batch_size=self.args.batch_size,collate_fn = collate_f)
        # load from self.data_set.tag_df
        self.ids_loader = DataLoader(self.df_idx,batch_size = self.args.batch_size)

    def train(self):
        res = {}

        for i in range(self.args.nepochs): #s
            epoch_loss = 0
            self.feature_extractor.train()
            torch.autograd.set_grad_enabled(True)

            for id1,id2,l in self.train_data_loader:
                # id1 : query
                # id2 : ids of serv
                # l : matched 
                l = l.to(torch.float)
                text1 = [torch.tensor(i) for i in id1]
                text2 = self.data_set[id2]
                # extract a column of a dataframe
                # and tolist it  
                p = self.feature_extractor(text1,text2.tolist())
                if self.cuda:
                    l = l.cuda()
                loss = self.loss(p.squeeze(),l.squeeze())   
                epoch_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            print('epoch:{},loss:{}'.format(i,epoch_loss))
            

            
            self.feature_extractor.eval()
            torch.autograd.set_grad_enabled(False)

            topks = [1,5,10,15,20,25,30]
            p,r,f,n = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
            pred = {}

#             if i%3 != 0:
#                 continue
            
            for test_k in self.test_keys:
                text1 = [torch.LongTensor(self.data_set.pos[test_k][0])]
                pos_ids = list(self.data_set.pos[test_k][1])
                
                sims = []
                for _ids in self.ids_loader:
                    text2 = self.data_set[_ids.cpu().numpy()]
                    _id_pred =  self.feature_extractor(text1,text2.tolist())
                    sims.append(_id_pred.cpu())
                sims = torch.cat(sims,dim=0).view(-1)
                pred[test_k] = sims
                sims_sort = sims.argsort(dim = -1,descending=True)

                for tk in topks:
                    _p,_r,_f,_n = self.get_metrics(self.df_idx[sims_sort],pos_ids,tk) #s
                    p[tk].append(_p)
                    r[tk].append(_r)
                    f[tk].append(_f)
                    n[tk].append(_n)
            p = {k:np.mean(v) for k,v in p.items()}
            r = {k:np.mean(v) for k,v in r.items()}
            f = {k:np.mean(v) for k,v in f.items()}
            n = {k:np.mean(v) for k,v in n.items()}
            table = {'p':p,'r':r,'f':f,'n':n}
#             print(pd.DataFrame(table).T,flush = True)
#             print('ave_pre:{}\tave_rec:{}'.format(np.mean(p),np.mean(r)))
#             pickle.dump(pred,open('./pred_{}'.format(i),'wb'))
#             pickle.dump(self.test_record,open('./true_{}'.format(i),'wb'))
            res[i]=pd.DataFrame(table).mean().T

        pd.DataFrame(res).T.to_csv('./out/'+self.model_str+str(self.train_test_id)+'.csv')

