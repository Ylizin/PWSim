from transformers import AutoTokenizer, AutoModelForMaskedLM

import torch
from torch import nn

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
            
        
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model =  AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

        self.cu = torch.cuda.is_available()
        if self.cu:
            self.model = self.model.cuda()
            
    
    def forward(self,seq):
        """
            seq should be a sequence of natural words
            expected to be List[str]
            output: (bzs,padded_len,hidden_size)
        """
        inp = self.tokenizer(seq,return_tensors="pt",padding = True)
        mask = inp.attention_mask
        if self.cu:
            inp = {k:v.cuda() for k,v in inp.items()}
        return self.model(**inp,output_hidden_states=True).hidden_states[-1],mask
        