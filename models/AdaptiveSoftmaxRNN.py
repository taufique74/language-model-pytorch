import torch
import torch.nn as nn
import torch.nn.functional as F
from .AdaptiveSoftmax import AdaptiveLogSoftmaxWithLoss

class AdaptiveSoftmaxRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, emb_dropout=0.0, rnn_dropout=0.2, tail_dropout=0.5, cutoffs=[20000, 50000]):
        super(AdaptiveSoftmaxRNN, self).__init__()
        ntoken = ntoken
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.out_dropout = nn.Dropout(0.5)
#         if adaptive_input:
#             self.encoder = AdaptiveInput(ninp, ntoken, cutoffs, tail_drop=tail_dropout)
#         else:
        self.encoder = nn.Embedding(ntoken, ninp)
            
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=rnn_dropout)
        # self.decoder = nn.Linear(nhid, ntoken)
        self.decoder = AdaptiveLogSoftmaxWithLoss(nhid, ntoken, cutoffs=cutoffs, div_value=2.0, tail_drop=tail_dropout)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        
        # weight sharing as described in the paper
#         if tie_weights and adaptive_input:
#             for i in range(len(cutoffs)):
#                 self.encoder.tail[i][0].weight = self.decoder.tail[i][1].weight
              
#                 # sharing the projection layers
#                 self.encoder.tail[i][1].weight = torch.nn.Parameter(self.decoder.tail[i][0].weight.transpose(0,1))

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, targets):
        emb = self.emb_dropout(self.encoder(input)) # (seq_len, bsz, ninp)
        output, hidden = self.rnn(emb, hidden) # (seq_len, bsz, ninp)
        output = self.out_dropout(output)
        output = output.view(-1,output.size(2)) # (seq_len*bsz, ninp)
        # output = output.transpose(0,1)
        # targets = targets.view(targets.size(0) * targets.size(1)) # (seq_len * bsz)
        # targets = targets.transpose(0,1)
        output, loss = self.decoder(output, targets)
        return output, hidden, loss

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))