from .generic import GeneralEval
import logging
import torch
from torch import nn, optim
from models.LSTM.LSTM import LSTMModule
import numpy as np
from torch.utils.data import SubsetRandomSampler,DataLoader
from collections import defaultdict
from configs.globalConfigs import args as _args
import pickle

class LSTMEval(GeneralEval):
    def __init__(self,args=_args):
        super().__init__()
        self.args = args
        self.train_pos = torch.tensor(self.data_set.train_pos).to(torch.int)
        self.train_neg = torch.tensor(self.data_set.train_neg).to(torch.int)
        self.test_pos = torch.tensor(self.data_set.test_pos).to(torch.int)
        self.test_idxs = torch.tensor(self.data_set.total_idx).to(torch.int)
        self.cuda = args.cuda
        self.cos = nn.CosineSimilarity()
        self.train_pos = torch.cat(
            [self.train_pos,
            torch.ones((self.train_pos.shape[0],1)).to(torch.int)],
            dim = 1)
        self.train_neg = torch.cat([self.train_neg,torch.zeros((self.train_neg.shape[0],1)).to(torch.int)],dim = 1)
        # self.test_pos = torch.cat([self.test_pos,torch.ones((self.test_pos.shape[0],1)).to(torch.int)],dim = 1)
        self.init_fe()
        self.init_optim()
        self.init_loss()
        self.init_data_loader()

    def init_fe(self):
        self.feature_extractor = LSTMModule()

    def init_optim(self):
        self.optim = optim.Adam(self.feature_extractor.parameters(),self.args.lr,weight_decay = self.args.weight_decay)

    def init_loss(self):
        self.loss = nn.MSELoss()    

    def init_data_loader(self):
        self.train_idxes =np.concatenate([self.train_pos,self.train_neg])
        np.random.shuffle(self.train_idxes)
        # self.train_sampler = SubsetRandomSampler(self.train_idxes)
        class ds:
            def __init__(self,train_idxes):
                self.train_idxes = train_idxes
            def __len__(self):
                return len(self.train_idxes)
            def __getitem__(self,_id):
                return self.train_idxes[_id,0],self.train_idxes[_id,1],self.train_idxes[_id,2]
        self.train_data_loader = DataLoader(ds(self.train_idxes),batch_size=self.args.batch_size)
        self.ids_loader = DataLoader(range(self.data_set.total_idx),batch_size = self.args.batch_size)
        self.test_record = defaultdict(list)
        for k,v in self.test_pos:
            self.test_record[k.item()].append(v.item())

    def train(self):
        for i in range(self.args.nepochs): #s
            epoch_loss = 0
            self.feature_extractor.train()
            for id1,id2,l in self.train_data_loader:
                l = l.to(torch.float)
                text1 = self.data_set[id1]
                text2 = self.data_set[id2]
                # extract a column of a dataframe
                # and tolist it  
                f1 = torch.mean(torch.stack([self.feature_extractor(t[1].tolist()) for t in text1.items()],dim=-1),dim=-1)
                f2 = torch.mean(torch.stack([self.feature_extractor(t[1].tolist()) for t in text2.items()],dim=-1),dim=-1)
                if self.cuda:
                    l = l.cuda()
                p = self.cos(f1,f2)
                loss = self.loss(p,l)   
                epoch_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            print('epoch:{},loss:{}'.format(i,loss))
            self.feature_extractor.eval()
            torch.autograd.set_grad_enabled(False)

            all_f = []
            for _ids in self.ids_loader:
                _t = self.data_set[_ids]
                _f = torch.mean(torch.stack([self.feature_extractor(t[1].tolist()) for t in _t.items()],dim=-1),dim=-1)
                all_f.append(_f.cpu())
            all_f = torch.cat(all_f,dim = 0).view(self.data_set.total_idx,-1)

            p,r = [],[]
            pred = {}

            for test_id,pos_ids in self.test_record.items():
                if len(pos_ids)== 1:
                    # pass case with only diagonal 
                    continue
                test_t = self.data_set[[test_id]]
                test_f = torch.mean(torch.stack([self.feature_extractor(t[1].tolist()) for t in test_t.items()],dim=-1),dim=-1).cpu()
                sims = self.cos(test_f,all_f)
                pred[test_id] = sims

                sims_sort = sims.argsort(dim = -1,descending=True)
                _p,_r = self.get_metrics(sims_sort,pos_ids,self.args.topk) #s
                p.append(_p)
                r.append(_r)
            print('ave_pre:{}\tave_rec:{}'.format(np.mean(p),np.mean(r)))
            #pickle.dump(pred,open('./pred_{}'.format(i),'wb'))
            #pickle.dump(self.test_record,open('./true_{}'.format(i),'wb'))
            torch.autograd.set_grad_enabled(True)
