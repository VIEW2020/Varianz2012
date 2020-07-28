'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import sys
sys.path.append('../lib/')

import numpy as np
import pandas as pd
import feather
import pickle as pkl

import torch
import torch.utils.data as utils
import torch.optim as optim
import torch.nn.functional as F

from deep_survival import *
from rnn_models import *
from utils import *
from hyperparameters import Hyperparameters

import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from captum.attr import configure_interpretable_embedding_layer

from pdb import set_trace as bp


def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code_' + hp.gender + '.feather')
    num_embeddings = df_index_code.shape[0]
    
    print('Concat train/val/test data...')
    x = np.concatenate((data['x_trn'], data['x_val'], data['x_tst']))
    codes = np.concatenate((data['codes_trn'], data['codes_val'], data['codes_tst']))
    month = np.concatenate((data['month_trn'], data['month_val'], data['month_tst']))
    diagt = np.concatenate((data['diagt_trn'], data['diagt_val'], data['diagt_tst']))
    
    print('Create data loader...')
    batch_size = 50
    data_tensors = utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(codes), torch.from_numpy(month), torch.from_numpy(diagt))
    data_loader = utils.DataLoader(data_tensors, batch_size = batch_size, shuffle = False,  drop_last = False)

    #######################################################################################################

    # Neural Net
    net = NetRNN_Interpret(x.shape[1], num_embeddings+1, hp).to(hp.device) #+1 for zero padding
    net.load_state_dict(torch.load(hp.log_dir + hp.test_model, map_location=hp.device))
    net.eval()

    # Captum
    interpretable_embed_codes = configure_interpretable_embedding_layer(net, 'embed_codes')
    interpretable_embed_diagt = configure_interpretable_embedding_layer(net, 'embed_diagt')
    torch.backends.cudnn.enabled = False # this is necessary for the backpropagation of RNNs models in eval mode
    ig = IntegratedGradients(net)

    # Attributions for each person
    x_attributions = np.zeros_like(x, dtype=float)
    codes_attributions = np.zeros_like(codes, dtype=float)
    
    with torch.no_grad():
        for batch_idx, (x_b, codes_b, month_b, diagt_b) in enumerate(tqdm(data_loader)):
            x_b = x_b.to(hp.device)
            codes_b = codes_b.to(hp.device)
            month_b = month_b.to(hp.device)
            diagt_b = diagt_b.to(hp.device)
            
            seq_length = (codes_b>0).sum(dim=-1)
            codes_embedding = interpretable_embed_codes.indices_to_embeddings(codes_b.long())
            diagt_embedding = interpretable_embed_diagt.indices_to_embeddings(diagt_b.long())

            data = (x_b, codes_embedding, month_b, diagt_embedding)
            data_baseline = tuple([torch.zeros_like(data[i]) for i in range(len(data))])
            
            attributions = ig.attribute(inputs=data, baselines=data_baseline, additional_forward_args=seq_length, n_steps=30)
            x_attributions[(batch_idx*batch_size):min((batch_idx+1)*batch_size, x.shape[0])] = attributions[0].detach().cpu().numpy()
            codes_attributions[(batch_idx*batch_size):min((batch_idx+1)*batch_size, x.shape[0])] = attributions[1].sum(dim=-1).detach().cpu().numpy()

    # Save
    np.savez(hp.data_pp_dir + 'integrated_gradients_' + hp.gender + '.npz', x_attributions=x_attributions, codes_attributions=codes_attributions)


if __name__ == '__main__':
    main()
