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

class AVEEval(GeneralEval):
    def __init__(self,args=_args):
        super().__init__()
        self.args = args
        self.cuda = args.cuda
        self.cos = nn.CosineSimilarity()
        self.train_pairs = self.data_set.train_pairs    
        self.test_keys = self.data_set.test_keys
        self.df_idx = self.data_set.df_idx
        self.sig = nn.Sigmoid()
        self.init_fe()
        self.init_optim()
        self.init_loss()
        self.init_data_loader()
        self.init_att()
        
    def init_att(self):
        self.sfm = nn.Softmax(dim = -1)
        self.att_weight = nn.Linear(self.embedding_dim,self.embedding_dim,bias = False)
        if self.cuda:
            self.sfm = self.sfm.cuda()
            self.att_weight = self.att_weight.cuda()
        
    def att(self,sent1,sent2):
        '''
            assume tensor1, tensor2 are list of tensors with different length
        '''
        tensor1 = [self.emb(torch.LongTensor(i)) for i in sent1]
        tensor2 = [self.emb(i) for i in sent2]
        idfs2 = [self.idf(i) for i in sent2]
        
        inp1 = torch.nn.utils.rnn.pad_sequence(tensor1,batch_first=True)
        inp2 = torch.nn.utils.rnn.pad_sequence(tensor2,batch_first=True)
        idfs2 = torch.nn.utils.rnn.pad_sequence(idfs2,batch_first=True)
        mask1 = torch.sum(inp1,dim = -1).bool()
        len1 = torch.sum(mask1,dim = -1,keepdim=True).to(torch.float)
        mask2 = torch.sum(inp2,dim = -1).bool()
        len2 = torch.sum(mask2,dim = -1,keepdim=True).to(torch.float)
        weight = torch.matmul(self.att_weight(inp1),inp2.permute(0,2,1))
        sent2_w = self.sfm(torch.sum(weight,dim =-2).masked_fill(~mask2,-float('inf'))).unsqueeze(-1)
        return torch.sum(inp1,dim = -2)/len1,torch.sum(torch.mul(sent2_w,inp2),dim = -2)
        
    
    def init_fe(self):
        self.feature_extractor = LSTMModule()
        self.li = nn.Linear(self.feature_extractor.hidden_size*4,1)
        self.emb = self.feature_extractor.embedding
        self.idf = self.emb.get_idf
        self.embedding_dim = self.emb.embedding_dim
        if self.cuda:
            self.li = self.li.cuda()
            

    def init_optim(self):
        self.optim = optim.Adam(self.emb.parameters(),self.args.lr,weight_decay = self.args.weight_decay)

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
        self.ids_loader = DataLoader(self.df_idx,batch_size = 1*self.args.batch_size)

    def train(self):
        for i in range(self.args.nepochs): #s
            epoch_loss = 0
            self.feature_extractor.train()
            for id1,id2,l in self.train_data_loader:
                # id1 : query
                # id2 : ids of serv
                # l : matched 
                l = l.to(torch.float)
                f1 = id1
                text2 = self.data_set[id2]
                # extract a column of a dataframe
                # and tolist it  
                f2 = text2.tolist()
                #f2 = self.feature_extractor(text2.tolist())
                #f1 = self.feature_extractor(text1)
                
                # f1 = torch.mean(torch.stack([self.feature_extractor(t[1].tolist()) for t in text1.items()],dim=-1),dim=-1)
                # f2 = torch.mean(torch.stack([self.feature_extractor(t[1].tolist()) for t in text2.items()],dim=-1),dim=-1)
                if self.cuda:
                    l = l.cuda()
                #p = self.sig(self.li(torch.cat([f1,f2],dim = -1)))
                f1,f2 = self.att(f1,f2)
                p = self.cos(f1,f2)
                loss = self.loss(p,l)   
                epoch_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            print('epoch:{},loss:{}'.format(i,epoch_loss))
            
            
            if i%3 != 1:
                continue
                
            self.feature_extractor.eval()
            torch.autograd.set_grad_enabled(False)

            topks = [1,5,10,15,20,25,30]
            p,r,f,n = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
            pred = {}

            for test_k in self.test_keys:
                text1 = [torch.LongTensor(self.data_set.pos[test_k][0])]
                pos_ids = list(self.data_set.pos[test_k][1])
                k_f = [self.emb(i) for i in text1]
                test_f = torch.mean(k_f[0],dim = -2).view(1,-1).cpu()
                
                all_f = []
                for _ids in self.ids_loader:
                    _t = self.data_set[_ids.cpu().numpy()]
                    _,_f = self.att(text1,_t)
                    all_f.append(_f.cpu())
                all_f = torch.cat(all_f,dim = 0).view(len(self.data_set),-1)
                sims = self.cos(test_f,all_f).cpu()
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
            print(pd.DataFrame(table).T)
            # print('ave_pre:{}\tave_rec:{}'.format(np.mean(p),np.mean(r)))
            # pickle.dump(pred,open('./pred_{}'.format(i),'wb'))
            # pickle.dump(self.test_record,open('./true_{}'.format(i),'wb'))
            torch.autograd.set_grad_enabled(True)
