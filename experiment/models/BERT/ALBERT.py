from transformers import AlbertTokenizer, AlbertModel,BatchEncoding
import torch
from torch import nn

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
            
        
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.model =  AlbertModel.from_pretrained('albert-base-v2', return_dict=True)  
        self.cu = torch.cuda.is_available()
        if self.cu:
            self.model.cuda()
            
    
    def forward(self,seq):
        """
            seq should be a sequence of natural words
            expected to be List[str]
            output: (bzs,padded_len,hidden_size)
        """
        inp = self.tokenizer(seq,return_tensors="pt",padding = True)
        if self.cu:
            inp = BatchEncoding({k:v.cuda() for k,v in inp.items()})
        return self.model(**inp,output_hidden_states=True).last_hidden_state,inp.attention_mask
        