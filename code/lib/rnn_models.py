'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import numpy as np
import math
import pandas as pd
import pickle as pkl
import torch
import torch.utils.data as utils
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import LSTM, GRU
from tqdm import tqdm

from pdb import set_trace as bp


class NetRNN(nn.Module):
    def __init__(self, n_input, num_embeddings, hp):
        super(NetRNN, self).__init__()
        #Embedding layers
        self.embed_codes = nn.Embedding(num_embeddings = num_embeddings,   embedding_dim = hp.embedding_dim, padding_idx = 0)
        #self.embed_month = nn.Embedding(num_embeddings = hp.num_months_hx, embedding_dim = hp.embedding_dim, padding_idx = 0)        
        self.embed_diagt = nn.Embedding(num_embeddings = 4, embedding_dim = hp.embedding_dim, padding_idx = 0)
        # RNN
        self.embedding_dim = hp.embedding_dim #+ 1
        self.num_layes = 1
        self.rnn = LSTM(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = self.num_layes, batch_first = True, dropout = 0, bidirectional = True)
        # Fully connected layers
        self.relu = nn.ReLU()
        self.fc_size = n_input + self.embedding_dim
        self.fc0 = nn.Linear(self.fc_size, self.fc_size)
        self.fc1 = nn.Linear(self.fc_size, self.fc_size)
        self.fc2 = nn.Linear(self.fc_size, 1, bias=False)

    def forward(self, x, code, month, diagt, time=None):
        if time is not None:
            x = torch.cat([x, time], 1)
        embedded_codes = self.embed_codes(code.long())
        #embedded_month = self.embed_month(month.long())
        embedded_diagt = self.embed_diagt(diagt.long())
        #embedded = torch.cat((embedded_codes, month.float().unsqueeze(dim=-1)), dim=-1) #+ embedded_diagt
        embedded = embedded_codes + embedded_diagt
        seq_length = (code>0).sum(dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_length.clamp(min=1), batch_first=True, enforce_sorted=False)
        output, (hidden, _) = self.rnn(packed)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output.view(-1, seq_length.max(), 2, self.embedding_dim) # view(batch, seq_len, num_directions, hidden_size)
        #output = output.sum(dim=-2).sum(dim=-2)
        output, _ = output.max(dim=-2)
        output, _ = output.max(dim=-2)
        #output = output/(2*seq_length.clamp(min=1).unsqueeze(dim=-1))
        output.masked_fill_((seq_length == 0).view(-1, 1), 0)
        
        #hidden = hidden.view(self.num_layes, 2, -1, self.embedding_dim)[-1] # view(num_layers, num_directions, batch, hidden_size)
        #hidden.masked_fill_((seq_length == 0).view(-1, 1), 0)
        #x = torch.cat((x, hidden[0] + hidden[1]), dim=-1)
        
        x = torch.cat((x, output), dim=-1)
        x = x + self.relu(self.fc0(x)) # skip connections
        x = x + self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
