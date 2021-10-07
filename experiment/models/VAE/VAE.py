import numpy as np
import torch
from torch import nn
from configs.VAEConfigs import args as _args    
import pickle


cos = nn.CosineSimilarity()

def vae_loss(X_BoW, predict_X_BoW,mu,log_var):
#     mse_loss = torch.nn.functional.mse_loss(X_BoW,predict_X_BoW)
    mse_loss = torch.nn.functional.binary_cross_entropy_with_logits(X_BoW, predict_X_BoW,reduction = 'sum')
    KLD_element = mu.pow(2).add(log_var.exp()).mul(-1).add(1).add(log_var)
    KLD = torch.sum(KLD_element).mul(-0.5)
    
    alpha = 1.0
    
    return mse_loss + KLD *alpha

class VAE(nn.Module):
    def __init__(self,args=_args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.pred_size = 1000

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self._cuda = args.cuda
        self.idf_path = args.idf_path
        self.vae_path = args.vae_path
        
        # embedding_size is a dimension of a word2embedding matrix
        self.f_bow = nn.Linear(args.vocab_size,args.embedding_size,bias = False)
        self.f_pi = nn.Linear(args.embedding_size,args.embedding_size)
        self.encoder = nn.Sequential(self.f_bow,self.relu,self.f_pi)
        self.f_mu = nn.Linear(args.embedding_size, args.topic_size)
        self.f_sigma = nn.Linear(args.embedding_size, args.topic_size)
        self.f_theta = nn.Linear(args.topic_size, args.topic_size)
        # for predict tag
        self.f_phi = nn.Linear(args.topic_size,self.vocab_size)
        self.bn = nn.BatchNorm1d(args.topic_size)

        
        if args.cuda:
            self = self.cuda()

            


    def vectorize_pred_bow(self,bow):
        len_batch = len(bow)
        stacked_bow = []
        
        if False:
            stacked_bow = bow
        else:
            for sent in bow:
                # id_f = torch.tensor(idx_freq)
                # idx = id_f[:,0]
                # pend_zeros = torch.zeros_like(idx)
                # idx = torch.stack([pend_zeros,idx])
                # v = id_f[:,1].to(torch.float)
                # tensor_bow = torch.sparse.FloatTensor(idx,v,torch.Size([1,self.vocab_size])).to_dense()
                tensor_bow = torch.zeros(self.vocab_size)
                for idx, freq in sent:
                    tensor_bow[idx] = freq
                stacked_bow.append(tensor_bow)
        stacked_bow = torch.stack(stacked_bow).view(-1,self.vocab_size)
        if self._cuda:
            stacked_bow = stacked_bow.cuda()

        return stacked_bow#.mul(self.idf.view(1,-1))        
    
    def vectorize_bow(self,bow):
        len_batch = len(bow)
        stacked_bow = []
        
        if False:
            stacked_bow = bow
        else:
            for sent in bow:
                # id_f = torch.tensor(idx_freq)
                # idx = id_f[:,0]
                # pend_zeros = torch.zeros_like(idx)
                # idx = torch.stack([pend_zeros,idx])
                # v = id_f[:,1].to(torch.float)
                # tensor_bow = torch.sparse.FloatTensor(idx,v,torch.Size([1,self.vocab_size])).to_dense()
                tensor_bow = torch.zeros(self.vocab_size)
                for idx, freq in sent:
                    tensor_bow[idx] = freq
                stacked_bow.append(tensor_bow)
        stacked_bow = torch.stack(stacked_bow).view(-1,self.vocab_size)
        if self._cuda:
            stacked_bow = stacked_bow.cuda()

        return stacked_bow#.mul(self.idf.view(1,-1))

    def reparameterize(self, mu, log_var):
        #std is the standard deviation , is the sigma
        std = log_var.mul(0.5).exp()
        if self.cuda:
            eps = torch.normal(mu,std).cuda()
        else:
            eps = torch.normal(mu,std)
        return eps

    
    def forward(self,X_Bows,training=True,pred = None):
        """
        forward rebuilt the bows
        
        :param X_Bows: input bows
        :type X_Bows: tensor:[bzs,vocab_size]
        """
        X_Bows = self.vectorize_bow(X_Bows)
        X_Bows = self.softmax(X_Bows)
        _X_BoWs = self.encoder(X_Bows)
        pi = self.relu(_X_BoWs)
        mu = self.f_mu(pi)
#         mu = self.bn(self.f_mu(pi))
        log_var = self.f_sigma(pi)
        z = self.reparameterize(mu, log_var)
        theta = self.relu(self.f_theta(z))
        theta = self.softmax(theta)
        out_bow = self.softmax(self.f_phi(theta)) # nothing changed 
        if not pred is None:
            X_Bows = self.softmax(self.vectorize_pred_bow(pred))
        loss = None
#         loss= vae_loss(X_Bows,out_bow,mu,log_var)
        if training:
            return out_bow, theta, loss, mu, log_var, X_Bows
        else:
            return theta,X_Bows
    
    
    

