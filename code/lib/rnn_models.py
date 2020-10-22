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
from torch.nn import LSTM, GRU, ConstantPad1d
from tqdm import tqdm
from attention_models import Attention

from pdb import set_trace as bp


def flip_batch(x, seq_length):
    assert x.shape[0] == seq_length.shape[0], 'Dimension Mismatch!'
    for i in range(x.shape[0]):
        x[i, :seq_length[i]] = x[i, :seq_length[i]].flip(dims=[0])
    return x


class NetRNN(nn.Module):
    def __init__(self, num_input, num_embeddings, hp):
        super().__init__()
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
            self.pad_fw = ConstantPad1d((1, 0), 0.)
            self.pad_bw = ConstantPad1d((0, 1), 0.)
        if self.rnn_type == 'LSTM':
            self.rnn_fw = LSTM(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = self.num_rnn_layers, batch_first = True, dropout = hp.dropout, bidirectional = False)
            self.rnn_bw = LSTM(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = self.num_rnn_layers, batch_first = True, dropout = hp.dropout, bidirectional = False)
        else:
            self.rnn_fw =  GRU(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = self.num_rnn_layers, batch_first = True, dropout = hp.dropout, bidirectional = False)
            self.rnn_bw =  GRU(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = self.num_rnn_layers, batch_first = True, dropout = hp.dropout, bidirectional = False)
        if self.summarize == 'output_attention':
            self.attention_fw = Attention(embedding_dim = self.embedding_dim)
            self.attention_bw = Attention(embedding_dim = self.embedding_dim)
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
        if self.add_month == 'concat':
            month = month/float(self.num_months_hx)
            delta = torch.clamp(month[:,1:]-month[:,:-1], min=0)
            delta_fw = self.pad_fw(delta)
            delta_bw = self.pad_bw(delta)
            embedded_fw = torch.cat((embedded, delta_fw.unsqueeze(dim=-1)), dim=-1)
            embedded_bw = torch.cat((embedded, delta_bw.unsqueeze(dim=-1)), dim=-1)
            embedded_bw = flip_batch(embedded_bw, seq_length)
        else:
            embedded_fw = embedded
            embedded_bw = flip_batch(embedded, seq_length)
        # RNN #############################################################################################################################
        packed_fw = nn.utils.rnn.pack_padded_sequence(embedded_fw, seq_length.clamp(min=1), batch_first = True, enforce_sorted = False)
        packed_bw = nn.utils.rnn.pack_padded_sequence(embedded_bw, seq_length.clamp(min=1), batch_first = True, enforce_sorted = False)
        if self.rnn_type == 'LSTM':
            output_fw, (hidden_fw, _) = self.rnn_fw(packed_fw)
            output_bw, (hidden_bw, _) = self.rnn_bw(packed_bw)
        elif self.rnn_type == 'GRU':
            output_fw, hidden_fw = self.rnn_fw(packed_fw)
            output_bw, hidden_bw = self.rnn_bw(packed_bw)
        if self.summarize == 'hidden':
            hidden_fw = hidden_fw[-1] # view(num_layers, num_directions=1, batch, hidden_size)[last_state]
            hidden_bw = hidden_bw[-1] # view(num_layers, num_directions=1, batch, hidden_size)[last_state]
            summary_0, summary_1 = hidden_fw, hidden_bw
        else:
            output_fw, _ = nn.utils.rnn.pad_packed_sequence(output_fw, batch_first=True)
            output_bw, _ = nn.utils.rnn.pad_packed_sequence(output_bw, batch_first=True)
            output_fw = output_fw.view(-1, max(1, seq_length.max()), self.embedding_dim) # view(batch, seq_len, num_directions=1, hidden_size)
            output_bw = output_bw.view(-1, max(1, seq_length.max()), self.embedding_dim) # view(batch, seq_len, num_directions=1, hidden_size)
            if self.summarize == 'output_max':
                output_fw, _ = output_fw.max(dim=1)
                output_bw, _ = output_bw.max(dim=1)
                summary_0, summary_1 = output_fw, output_bw
            elif self.summarize == 'output_sum':
                output_fw = output_fw.sum(dim=1)
                output_bw = output_bw.sum(dim=1)
                summary_0, summary_1 = output_fw, output_bw
            elif self.summarize == 'output_avg':
                output_fw = output_fw.sum(dim=1)/(seq_length.clamp(min=1).view(-1, 1))
                output_bw = output_bw.sum(dim=1)/(seq_length.clamp(min=1).view(-1, 1))
                summary_0, summary_1 = output_fw, output_bw
            elif self.summarize == 'output_attention':
                mask = (code>0)[:, :max(1, seq_length.max())]
                summary_0, _ = self.attention_fw(output_fw, mask)
                summary_1, _ = self.attention_bw(output_bw, mask)
            
        # Fully connected layers ##########################################################################################################    
        x = torch.cat((x, summary_0, summary_1), dim=-1)
        x = self.mlp(x)
        return x


class NetRNNFinal(nn.Module):
    def __init__(self, num_input, num_embeddings, hp):
        super().__init__()
        # Parameters ######################################################################################################################
        self.num_months_hx = hp.num_months_hx-1
        self.num_rnn_layers = hp.num_rnn_layers
        self.embedding_dim = hp.embedding_dim
        # Embedding layers ################################################################################################################
        self.embed_codes = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = hp.embedding_dim, padding_idx = 0)
        self.embed_diagt = nn.Embedding(num_embeddings = 5, embedding_dim = hp.embedding_dim, padding_idx = 0)
        # RNN #############################################################################################################################
        self.embedding_dim = self.embedding_dim + 1
        self.pad_fw = ConstantPad1d((1, 0), 0.)
        self.rnn_fw =  GRU(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = self.num_rnn_layers, batch_first = True, dropout = hp.dropout, bidirectional = False)
        self.attention_fw = Attention(embedding_dim = self.embedding_dim)
        # Fully connected layers ##########################################################################################################
        fc_size = num_input + self.embedding_dim
        layers = []
        layers.append(nn.Linear(fc_size, fc_size))
        layers.append(nn.ELU())
        layers.append(nn.Linear(fc_size, 1))
        self.mlp = nn.Sequential(*layers)        

    def forward(self, x, code, month, diagt, time=None, seq_length=None):
        if seq_length is None:
            seq_length = (code>0).sum(dim=-1)
        # Embedding layers ################################################################################################################
        embedded = self.embed_codes(code.long())
        embedded = embedded + self.embed_diagt(diagt.long())    
        month = month/float(self.num_months_hx)
        delta = torch.clamp(month[:,1:]-month[:,:-1], min=0)
        delta_fw = self.pad_fw(delta)
        embedded_fw = torch.cat((embedded, delta_fw.unsqueeze(dim=-1)), dim=-1)
        # RNN #############################################################################################################################
        packed_fw = nn.utils.rnn.pack_padded_sequence(embedded_fw, seq_length.clamp(min=1), batch_first = True, enforce_sorted = False)
        output_fw, _ = self.rnn_fw(packed_fw)
        output_fw, _ = nn.utils.rnn.pad_packed_sequence(output_fw, batch_first=True)
        output_fw = output_fw.view(-1, max(1, seq_length.max()), self.embedding_dim) # view(batch, seq_len, num_directions=1, hidden_size)
        summary_0, _ = output_fw.max(dim=1)
        # Fully connected layers ##########################################################################################################    
        x = torch.cat((x, summary_0), dim=-1)
        x = self.mlp(x)
        return x

