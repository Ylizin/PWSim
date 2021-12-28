from .tag_generic import GeneralEval
import logging
import torch
from torch import nn, optim
from models.LSTM.LSTM import LSTMModule
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
        
        self.model_str = "TagLSTMEval"
        self.use_ext_query = args.use_ext_query

        self.init_fe()
        self.init_optim()
        self.init_loss()
        self.init_data_loader()

    def init_fe(self):
        self.feature_extractor = LSTMModule(suffix = str(self.train_test_id))
        self.emb = self.feature_extractor.embedding
        self.relu = nn.ReLU()


    def init_optim(self):
        self.optim = optim.Adam(self.feature_extractor.parameters(),self.args.lr,weight_decay = self.args.weight_decay)

    def init_loss(self):
#         self.loss = nn.MSELoss()    
        self.loss = nn.BCELoss()

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
            for id1,id2,l in self.train_data_loader:
                # id1 : query
                # id2 : ids of serv
                # l : matched 
                l = l.to(torch.float)
                text2 = self.data_set[id2]
                # extract a column of a dataframe
                if self.use_ext_query:
                    # query 向量， 使用ext之后的tag作为query
                    f1 = self.feature_extractor([torch.LongTensor(s) for s in self.query_syn(id1)])    # the query using ext texts as raw text
                else:
                    # query 向量， 使用原始的tag作为query
                    f1 = self.feature_extractor([torch.LongTensor(s) for s in id1])


                # f1 = torch.stack([torch.mean(self.emb(torch.LongTensor(i)),dim = 0) for i in id1],dim=0).cuda()
                f2 = self.feature_extractor(text2.tolist())
                # f1 = torch.mean(torch.stack([self.feature_extractor(t[1].tolist()) for t in text1.items()],dim=-1),dim=-1)
                # f2 = torch.mean(torch.stack([self.feature_extractor(t[1].tolist()) for t in text2.items()],dim=-1),dim=-1)
                if self.cuda:
                    l = l.cuda()
                p = self.relu(self.cos(f1,f2)) # map to [0,1]
                loss = self.loss(p,l)   
                epoch_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            print('epoch:{},loss:{}'.format(i,epoch_loss))
            self.feature_extractor.eval()
            torch.autograd.set_grad_enabled(False)

            all_f = []
            for _ids in self.ids_loader:
                _t = self.data_set[_ids.cpu().numpy()]
                _f = self.feature_extractor(_t)

                all_f.append(_f.cpu())
            all_f = torch.cat(all_f,dim = 0).view(len(self.data_set),-1)
            
            topks = [1,5,10,15,20]
            p,r,f,n = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
            pred = {}

            for test_k in self.test_keys:
                if self.use_ext_query:
                    text1 = [torch.LongTensor(self.query_syn([self.data_set.pos[test_k][0]]))[0]]
                else:
                    text1 = [torch.LongTensor(self.data_set.pos[test_k][0])]
                pos_ids = list(self.data_set.pos[test_k][1])
                test_f = self.feature_extractor(text1).cpu()
                sims = self.relu(self.cos(test_f,all_f))
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
            print(table)
            res[i]=pd.DataFrame(table).mean().T
            #print('ave_pre:{}\tave_rec:{}'.format(np.mean(p),np.mean(r)))
            # pickle.dump(pred,open('./lstm_pred','wb'))
            # pickle.dump(self.test_record,open('./true_{}'.format(i),'wb'))
            torch.autograd.set_grad_enabled(True)
        pd.DataFrame(res).T.to_csv('./out/'+self.model_str+str(self.train_test_id)+'.csv')
