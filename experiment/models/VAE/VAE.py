import numpy as np
import torch
from torch import nn

cos = nn.CosineSimilarity()

def vae_loss(X_BoW, predict_X_BoW,mu,log_var):
    mse_loss = torch.sum(1-cos(X_BoW,predict_X_BoW))
    KLD_element = mu.pow(2).add(log_var.exp()).mul(-1).add(1).add(log_var)
    KLD = torch.sum(KLD_element).mul(-0.5)
    
    alpha = 0.3
    
    return mse_loss + KLD *alpha

class VAE(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.cuda = args.cuda

        self.f_bow = nn.Linear(args.vocab_size,args.embedding_size,bias = False)
        self.f_pi = nn.Linear(args.embedding_size,args.embedding_size)
        self.encoder = nn.Sequential(self.f_bow,self.relu,self.pi)
        self.f_mu = nn.Linear(args.embedding_size, args.topic_size)
        self.f_sigma = nn.Linear(args.embedding_size, args.topic_size)
        self.f_theta = nn.Linear(args.topic_size, args.topic_size)
        self.f_phi = nn.Linear(args.topic_size, args.vocab_size)

    def reparameterize(self, mu, log_var):
        #std is the standard deviation , is the sigma
        #
        std = log_var.mul(0.5).exp()
        if self.cuda:
            eps = torch.normal(mu,std).cuda()
        else:
            eps = torch.normal(mu,std)
        return eps

    def forward(self,X_Bows):
        """
        forward rebuilt the bows
        
        :param X_Bows: input bows
        :type X_Bows: tensor:[bzs,vocab_size]
        """
        X_Bows = self.softmax(X_Bows)
        _X_BoWs = self.encoder(X_Bows)
        pi = self.relu(_X_BoWs)
        mu = self.f_mu(pi)
        log_var = self.f_sigma(pi)
        z = self.reparameterize(mu, log_var)
        theta = self.relu(self.f_theta(z))
        theta = self.softmax(theta)
        out_bow = self.softmax(self.f_phi(theta))
        return out_bow, theta, mu, log_var, X_Bows
    
    

