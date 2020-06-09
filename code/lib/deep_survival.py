'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import numpy as np
import math
import pandas as pd
import torch
import torch.utils.data as utils
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_add
from torch_scatter.composite import scatter_softmax
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


class CoxPHLoss(torch.nn.Module):
    """
    Approximated Cox Proportional Hazards loss (negative partial likelihood 
        with 1 control sample)
    
    Args:
        risk_case: The predicted risk for people who experienced the event (batch_size,) 
        risk_control: The predicted risk for people who where at risk at the time 
            when the corresponding case experienced the event (batch_size,) 

    Returns:
        loss: Scalar loss
    """    
    def __init__(self):
        super(CoxPHLoss, self).__init__()
        
    def forward(self, risk_case, risk_control):
        loss = (F.softplus(risk_control - risk_case)).mean()
        return loss

    
def get_case_control(x_case, time_case, max_idx_control, code_case, month_case, diagt_case, x, code, month, diagt, hp):
    control_idx = (np.random.uniform(size=(x_case.shape[0],)) * max_idx_control.numpy()).astype(int)
    x_control = x[control_idx]
    x_cc = torch.cat([x_case, x_control]).to(hp.device)
    if hp.nonprop_hazards:
        time_cc = torch.unsqueeze(torch.cat([time_case, time_case]), 1).float().to(hp.device) # time is the same for cases and controls
    else:
        time_cc = None
    code_control = code[control_idx]
    code_cc = torch.cat([code_case, code_control]).to(hp.device)
    month_control = month[control_idx]
    month_cc = torch.cat([month_case, month_control]).to(hp.device)
    diagt_control = diagt[control_idx]
    diagt_cc = torch.cat([diagt_case, diagt_control]).to(hp.device)    
    return x_cc, time_cc, code_cc, month_cc, diagt_cc


def trn(trn_loader, x_trn, code_trn, month_trn, diagt_trn, model, criterion, optimizer, hp):
    model.train()
    for batch_idx, (x_case, time_case, max_idx_control, code_case, month_case, diagt_case) in enumerate(tqdm(trn_loader)):
        # Get controls
        x_cc, time_cc, code_cc, month_cc, diagt_cc = get_case_control(x_case, time_case, max_idx_control, code_case, month_case, diagt_case, x_trn, code_trn, month_trn, diagt_trn, hp)
        
        # Optimise
        optimizer.zero_grad()
        risk_case, risk_control = model(x_cc, code_cc, month_cc, diagt_cc, time_cc).chunk(2)
        loss = criterion(risk_case, risk_control)
        loss.backward()
        optimizer.step()

        
def val(val_loader, x_val, code_val, month_val, diagt_val, model, criterion, epoch, hp):
    loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x_case, time_case, max_idx_control, code_case, month_case, diagt_case) in enumerate(tqdm(val_loader)):
            # Get controls
            x_cc, time_cc, code_cc, month_cc, diagt_cc = get_case_control(x_case, time_case, max_idx_control, code_case, month_case, diagt_case, x_val, code_val, month_val, diagt_val, hp)
            
            # Compute Loss
            risk_case, risk_control = model(x_cc, code_cc, month_cc, diagt_cc, time_cc).chunk(2)
            loss += criterion(risk_case, risk_control)

        print('Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()/len(val_loader)))
        return loss


class NetAttention(nn.Module):
    def __init__(self, n_input, num_embeddings, hp):
        super(NetAttention, self).__init__()
        self.embedding_dim = hp.embedding_dim
        # Embedding layers
        self.embed_codes = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = hp.embedding_dim, padding_idx = 0, max_norm = 1)
        self.embed_diagt = nn.Embedding(num_embeddings = 5, embedding_dim = hp.embedding_dim, padding_idx = 0, max_norm = 1)
        # Positional encoding
        pos_encodings = torch.zeros(hp.num_months_hx, hp.embedding_dim)
        position = torch.arange(0, hp.num_months_hx, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hp.embedding_dim, 2).float() * (-math.log(10000.0) / hp.embedding_dim))
        pos_encodings[:, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 1::2] = torch.cos(position * div_term)
        self.embed_month = nn.Embedding.from_pretrained(pos_encodings, max_norm = 1)
        # Attention
        self.attention = Attention(embedding_dim = hp.embedding_dim)
        # Fully connected layers
        self.elu = nn.ELU()
        self.fc_size = n_input + hp.embedding_dim
        #self.fc0 = nn.Linear(self.fc_size, self.fc_size)
        #self.fc1 = nn.Linear(self.fc_size, self.fc_size)
        self.fc2 = nn.Linear(self.fc_size, 1, bias=False)

    def forward(self, x, code, month, diagt, time=None):
        if time is not None:
            x = torch.cat([x, time], 1)
        embedded_codes = self.embed_codes(code.long())
        embedded_month = self.embed_month(month.long())
        embedded_diagt = self.embed_diagt(diagt.long())
        summary, _ = self.attention(embedded_codes + embedded_month + embedded_diagt, code)
        x = torch.cat((x, summary), dim=-1)
        #x = x + self.elu(self.fc0(x)) # skip connections
        #x = x + self.elu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    

def log(model_name, concordance, brier, nbll, hp):
    df = pd.DataFrame({'model_name': model_name,
                       'np_seed': hp.np_seed,
                       'torch_seed': hp.torch_seed,
                       'min_count': hp.min_count,
                       'nonprop_hazards': hp.nonprop_hazards,
                       'batch_size': hp.batch_size,
                       'max_epochs': hp.max_epochs,
                       'patience': hp.patience,
                       'embedding_dim': hp.embedding_dim,
                       'num_months_hx': hp.num_months_hx,
                       'sample_comp_bh': hp.sample_comp_bh,
                       'concordance': concordance,
                       'brier': brier,
                       'nbll': nbll},
                       index=[0])
    with open(hp.data_dir + 'logfile.csv', 'a', newline='\n') as f:
        df.to_csv(f, mode='a', index=False, header=(not f.tell()))
                       
                       




