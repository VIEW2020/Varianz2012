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
from attention_models import Attention

from pdb import set_trace as bp


class NetRNN(nn.Module):
    def __init__(self, num_input, num_embeddings, hp):
        super(NetRNN, self).__init__()
        # Parameters ######################################################################################################################
        self.nonprop_hazards = hp.nonprop_hazards
        self.add_diagt = hp.add_diagt
        self.add_month = hp.add_month
        self.num_months_hx = hp.num_months_hx-1
        self.rnn_type = hp.rnn_type
        self.num_rnn_layers = hp.num_rnn_layers
        self.embedding_dim = hp.embedding_dim
        self.summarize = hp.summarize
        # Embedding layers ################################################################################################################
        self.embed_codes = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = hp.embedding_dim, padding_idx = 0)
        if self.add_month == 'embedding':
            self.embed_month = nn.Embedding(num_embeddings = hp.num_months_hx, embedding_dim = hp.embedding_dim, padding_idx = 0)
        if self.add_diagt:
            self.embed_diagt = nn.Embedding(num_embeddings = 5, embedding_dim = hp.embedding_dim, padding_idx = 0)
        # RNN #############################################################################################################################
        if self.add_month == 'concat':
            self.embedding_dim = self.embedding_dim + 1
        if self.rnn_type == 'LSTM':
            self.rnn = LSTM(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = self.num_rnn_layers, batch_first = True, dropout = hp.dropout, bidirectional = True)
        else:
            self.rnn =  GRU(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = self.num_rnn_layers, batch_first = True, dropout = hp.dropout, bidirectional = True)
        if self.summarize == 'output_attention':
            self.attention = Attention(embedding_dim = self.embedding_dim)
        # Fully connected layers ##########################################################################################################
        fc_size = num_input + 2*self.embedding_dim
        layers = []
        for i in range(hp.num_mlp_layers):
            layers.append(nn.Linear(fc_size, fc_size))
            layers.append(nn.ELU())
        layers.append(nn.Linear(fc_size, 1))
        self.mlp = nn.Sequential(*layers)        

    def forward(self, x, code, month, diagt, time=None, seq_length=None):
        if self.nonprop_hazards and (time is not None):
            x = torch.cat((x, time), dim=-1)
        if seq_length is None:
            seq_length = (code>0).sum(dim=-1)
        # Embedding layers ################################################################################################################
        embedded = self.embed_codes(code.long())
        if self.add_diagt:
            embedded = embedded + self.embed_diagt(diagt.long())        
        if self.add_month == 'embedding':
            embedded = embedded + self.embed_month(month.long())
        elif self.add_month == 'concat':
            embedded = torch.cat((embedded, (month/float(self.num_months_hx)).unsqueeze(dim=-1)), dim=-1)            
        # RNN #############################################################################################################################
        packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_length.clamp(min=1), batch_first = True, enforce_sorted = False)
        if self.rnn_type == 'LSTM':
            output, (hidden, _) = self.rnn(packed)
        elif self.rnn_type == 'GRU':
            output, hidden = self.rnn(packed)
        if self.summarize == 'hidden':
            hidden = hidden.view(self.num_rnn_layers, 2, -1, self.embedding_dim)[-1] # view(num_layers, num_directions, batch, hidden_size)[last_state]
            summary_0, summary_1 = hidden[0], hidden[1]
        else:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = output.view(-1, max(1, seq_length.max()), 2, self.embedding_dim) # view(batch, seq_len, num_directions, hidden_size)
            if self.summarize == 'output_max':
                output, _ = output.max(dim=1)
                summary_0, summary_1 = output[:,0,:], output[:,1,:]
            elif self.summarize == 'output_sum':
                output = output.sum(dim=1)
                summary_0, summary_1 = output[:,0,:], output[:,1,:]
            elif self.summarize == 'output_avg':
                output = output.sum(dim=1)/(seq_length.clamp(min=1).view(-1, 1, 1))
                summary_0, summary_1 = output[:,0,:], output[:,1,:]
            elif self.summarize == 'output_attention':
                output_0, output_1 = output[:,:,0,:], output[:,:,1,:]
                mask = (code>0)[:, :max(1, seq_length.max())]
                summary_0, _ = self.attention(output_0, mask)
                summary_1, _ = self.attention(output_1, mask)
            
        # Fully connected layers ##########################################################################################################    
        x = torch.cat((x, summary_0, summary_1), dim=-1)
        x = self.mlp(x)
        return x


class NetRNN_Minimal(nn.Module):
    def __init__(self, num_input, num_embeddings, hp):
        super(NetRNN_Minimal, self).__init__()
        # Parameters ######################################################################################################################
        self.num_months_hx = hp.num_months_hx-1
        self.embedding_dim = hp.embedding_dim
        # Embedding layers ################################################################################################################
        self.embed_codes = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = hp.embedding_dim, padding_idx = 0)
        self.embed_diagt = nn.Embedding(num_embeddings = 5, embedding_dim = hp.embedding_dim, padding_idx = 0)
        # RNN #############################################################################################################################
        self.embedding_dim = self.embedding_dim + 1
        self.rnn =  GRU(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = hp.num_rnn_layers, batch_first = True, dropout = hp.dropout, bidirectional = True)
        # Fully connected layers ##########################################################################################################
        fc_size = num_input + 2*self.embedding_dim
        layers = []
        layers.append(nn.Linear(fc_size, fc_size))
        layers.append(nn.ELU())
        layers.append(nn.Linear(fc_size, 1))
        self.mlp = nn.Sequential(*layers)        

    def forward(self, x, code, month, diagt, seq_length=None):
        if seq_length is None:
            seq_length = (code>0).sum(dim=-1)
        # Embedding layers ################################################################################################################
        embedded = self.embed_codes(code)
        embedded = embedded + self.embed_diagt(diagt)
        embedded = torch.cat((embedded, (month/float(self.num_months_hx)).unsqueeze(dim=-1)), dim=-1)            
        # RNN #############################################################################################################################
        packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_length.clamp(min=1), batch_first = True, enforce_sorted = False)
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output.view(-1, max(1, seq_length.max()), 2, self.embedding_dim) # view(batch, seq_len, num_directions, hidden_size)
        output, _ = output.max(dim=1)
        output.masked_fill_((seq_length == 0).view(-1, 1, 1), 0)
        summary_0, summary_1 = output[:,0,:], output[:,1,:]
        # Fully connected layers ##########################################################################################################    
        x = torch.cat((x, summary_0, summary_1), dim=-1)
        x = self.mlp(x)
        return x




