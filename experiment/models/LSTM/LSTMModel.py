import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence,pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.droprate = args.dropout
        self.bidirectional = args.bidirectional
        self.dropout = nn.Dropout(self.droprate)

        self.cuda = args.cuda
        self.layers = 1
        self.rnn = nn.LSTM(self.input_size,self.hidden_size,self.layers,bidirectional = args.bidirectional)

        if self.cuda:
            self.rnn = self.rnn.cuda()

    def forward(self,seqs_tensor:PackedSequence)->tuple:
        unsort_idx = seqs_tensor.unsorted_indices 
        if self.cuda:
            seqs_tensor.cuda()

        # lstm_output : (batch, seq_len, num_directions * hidden_size)
        # hn : (num_layers * num_directions, batch, hidden_size)
        # cn : (num_layers * num_directions, batch, hidden_size)
        lstm_output,(hn,cn) = self.rnn(seqs_tensor)
        # with pad_packed_sequence, no need to unsort lstm_output
        hn = hn.index_select(1,unsort_idx)
        cn = cn.index_select(1,unsort_idx)
        hn = hn.view(hn.size()[1],-1)
        cn = cn.view(cn.size()[1],-1)
        hn = self.dropout(hn)
        cn = self.dropout(cn)
        lstm_output = PackedSequence(self.dropout(lstm_output.data),lstm_output.batch_sizes,lstm_output.sorted_indices,lstm_output.unsorted_indices)# dropout on the .data
        out_tensor,lengths = pad_packed_sequence(lstm_output,batch_first = True)
        return (out_tensor,lengths),hn,cn