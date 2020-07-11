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
#from torch_scatter import scatter_add
#from torch_scatter.composite import scatter_softmax
from tqdm import tqdm

from pdb import set_trace as bp


class Attention(torch.nn.Module):
  """
  Dot-product attention module.
  
  Args:
    inputs: A `Tensor` with embeddings in the last dimension.
    mask: A `Tensor`. Dimensions are the same as inputs but without the embedding dimension.
      Values are 0 for 0-padding in the input and 1 elsewhere.
  Returns:
    outputs: The input `Tensor` whose embeddings in the last dimension have undergone a weighted average.
      The second-last dimension of the `Tensor` is removed.
    attention_weights: weights given to each embedding.
  """
  def __init__(self, embedding_dim):
    super(Attention, self).__init__()
    self.norm = np.sqrt(embedding_dim)
    self.context = nn.Parameter(torch.Tensor(embedding_dim)) # context vector
    self.linear_hidden = nn.Linear(embedding_dim, embedding_dim)
    self.reset_parameters()
    
  def reset_parameters(self):
    nn.init.normal_(self.context)

  def forward(self, input, mask):
    # Hidden representation of embeddings (no change in dimensions)
    hidden = torch.tanh(self.linear_hidden(input))
    # Compute weight of each embedding
    importance = torch.sum(hidden * self.context, dim=-1) / self.norm
    importance = importance.masked_fill(mask == 0, -1e9)
    # Softmax so that weights sum up to one
    attention_weights = F.softmax(importance, dim=-1)
    # Weighted sum of embeddings
    weighted_projection = input * torch.unsqueeze(attention_weights, dim=-1)
    # Output
    output = torch.sum(weighted_projection, dim=-2)
    return output, attention_weights


class AttentionWeighted(torch.nn.Module):
  """
  Dot-product attention module.
  
  Args:
    inputs: A `Tensor` with embeddings in the last dimension.
    mask: A `Tensor`. Dimensions are the same as inputs but without the embedding dimension.
      Values are 0 for 0-padding in the input and 1 elsewhere.
  Returns:
    outputs: The input `Tensor` whose embeddings in the last dimension have undergone a weighted average.
      The second-last dimension of the `Tensor` is removed.
    attention_weights: weights given to each embedding.
  """
  def __init__(self, embedding_dim):
    super(AttentionWeighted, self).__init__()
    self.norm = np.sqrt(embedding_dim)
    self.context = nn.Parameter(torch.Tensor(embedding_dim)) # context vector
    self.linear_hidden = nn.Linear(embedding_dim, embedding_dim)
    self.reset_parameters()
    
  def reset_parameters(self):
    nn.init.normal_(self.context)

  def forward(self, input, mask, weight):
    # Hidden representation of embeddings (no change in dimensions)
    hidden = torch.tanh(self.linear_hidden(input))
    # Compute weight of each embedding
    importance = torch.sum(hidden * self.context, dim=-1) / self.norm
    importance = importance*weight
    importance = importance.masked_fill(mask == 0, -1e9)
    # Softmax so that weights sum up to one
    attention_weights = F.softmax(importance, dim=-1)
    # Weighted sum of embeddings
    weighted_projection = input * torch.unsqueeze(attention_weights, dim=-1)
    # Output
    output = torch.sum(weighted_projection, dim=-2)
    return output, attention_weights


class ScatterAttention(torch.nn.Module):
  """
  Dot-product attention module.
  
  Args:
    input: A `Tensor` with embeddings in the last dimension.
    mask: A `Tensor`. Dimensions are the same as input but without the embedding dimension.
      Values are 0 for 0-padding in the input and 1 elsewhere.

  Returns:
    output: The input `Tensor` whose embeddings in the last dimension have undergone a weighted average.
      The second-last dimension of the `Tensor` is removed.
    attention_weights: weights given to each embedding.
  """
  def __init__(self, embedding_dim, index_dim):
    super(ScatterAttention, self).__init__()
    self.embedding_dim = embedding_dim
    self.index_dim = index_dim
    self.norm = np.sqrt(embedding_dim)
    self.context = nn.Parameter(torch.Tensor(embedding_dim)) # context vector
    self.linear_hidden = nn.Linear(embedding_dim, embedding_dim)
    self.reset_parameters()
    
  def reset_parameters(self):
    nn.init.normal_(self.context)

  def forward(self, input, mask, index):
    # Hidden representation of embeddings (no change in dimensions)
    hidden = torch.tanh(self.linear_hidden(input))
    # Compute weight of each embedding
    importance = torch.sum(hidden * self.context, dim=-1) / self.norm
    importance = importance.masked_fill(mask == 0, -1e9)
    # Softmax so that weights sum up to one
    attention_weights = scatter_softmax(importance, index, dim=-1)
    # Weighted sum of embeddings
    weighted_projection = input * torch.unsqueeze(attention_weights, dim=-1)
    # Output
    output = scatter_add(weighted_projection, torch.unsqueeze(index, dim=-1), dim=-2, dim_size=self.index_dim)
    return output, attention_weights


class NetAttention(nn.Module):
    def __init__(self, n_input, num_embeddings, hp):
        super(NetAttention, self).__init__()
        self.embedding_dim = hp.embedding_dim
        # Embedding layers
        self.embed_codes = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = hp.embedding_dim, padding_idx = 0, max_norm = 1, norm_type = 2.)
        # Exponential time decay coefficient
        self.decay = nn.Parameter(torch.zeros(1))
        # Attention
        self.attention = AttentionWeighted(embedding_dim = hp.embedding_dim)
        # Fully connected layers
        self.fc_size = n_input + hp.embedding_dim
        self.fc = nn.Linear(self.fc_size, 1, bias=False)

    def forward(self, x, code, month, diagt, time=None):
        if time is not None:
            x = torch.cat([x, time], 1)
        embedded_codes = self.embed_codes(code.long())
        decay = torch.abs(self.decay) #needs to be positive
        summary, _ = self.attention(embedded_codes, code, torch.exp(-decay*month))
        x = torch.cat((x, summary), dim=-1)
        x = self.fc(x)        
        return x


class NetTransformer(nn.Module):
    def __init__(self, n_input, num_embeddings, hp):
        super(NetTransformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.embedding_dim = hp.embedding_dim
        # Embedding layers
        self.embed_codes = nn.Embedding(num_embeddings = num_embeddings,   embedding_dim = hp.embedding_dim, padding_idx = 0)
        #self.embed_month = nn.Embedding(num_embeddings = hp.num_months_hx, embedding_dim = hp.embedding_dim, padding_idx = 0)        
        #self.embed_diagt = nn.Embedding(num_embeddings = 4,                embedding_dim = hp.embedding_dim, padding_idx = 0)
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model = hp.embedding_dim, nhead = 1, dim_feedforward = hp.embedding_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers = 3)
        # Fully connected layers
        self.relu = nn.ReLU()
        self.fc_size = n_input + hp.embedding_dim
        self.fc0 = nn.Linear(self.fc_size, self.fc_size)
        self.fc1 = nn.Linear(self.fc_size, self.fc_size)
        self.fc2 = nn.Linear(self.fc_size, 1, bias=False)

    def forward(self, x, code, month, diagt, time=None):
        if time is not None:
            x = torch.cat([x, time], 1)
        embedded_codes = self.embed_codes(code.long())
        #embedded_month = self.embed_month(month.long())
        #embedded_diagt = self.embed_diagt(diagt.long())
        mask = (code == 0)
        embedded = embedded_codes #+ embedded_month + embedded_diagt
        encoded = self.transformer_encoder(embedded.permute(1, 0, 2), src_key_padding_mask = mask).permute(1, 0, 2)
        encoded = encoded.sum(dim=-2) / ((~mask).sum(dim=-1, keepdim=True))
        #encoded, _ = encoded.max(dim=-2)
        encoded[torch.isnan(encoded)] = 0 # nan from encoder and division if all codes are 0/masked
        x = torch.cat((x, encoded), dim=-1)
        x = x + self.relu(self.fc0(x)) # skip connections
        x = x + self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

