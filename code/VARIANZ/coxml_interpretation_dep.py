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

from os import listdir

import statsmodels.stats.api as sms
from scipy import stats
import matplotlib.pyplot as plt

from pdb import set_trace as bp


def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code_' + hp.gender + '.feather')
    num_embeddings = df_index_code.shape[0]
    means = np.load(hp.data_pp_dir + 'means_' + hp.gender + '.npz')

    print('Add standard columns...')
    if hp.redundant_predictors:
        cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    else:
        cols_list = hp.reduced_col_list
    num_cols = len(cols_list)

    #######################################################################################################
        
    print('Compute HRs...')

    # Trained models
    if hp.redundant_predictors:
        tmp = listdir(hp.log_dir + 'all/')
        models = ['all/' + i for i in tmp if '.pt' in i]    
    else:
        tmp = listdir(hp.log_dir + 'all_no_redundancies/')
        models = ['all_no_redundancies/' + i for i in tmp if '.pt' in i]    

    log_hr_matrix = np.zeros((len(range(1, 6)), len(models)))

    # Neural Net
    num_input = num_cols+1 if hp.nonprop_hazards else num_cols
    net = NetRNNFinal(num_input, num_embeddings+1, hp).to(hp.device) #+1 for zero padding
    net.eval()

    for i in range(len(models)):
        print('HRs for model {}'.format(i))
        
        # Restore variables from disk
        net.load_state_dict(torch.load(hp.log_dir + models[i], map_location=hp.device))

        # Compute risk for all ages
        for j in tqdm(range(1, 6)):
            with torch.no_grad():
                x_b = torch.zeros((1, num_cols), device=hp.device)
                codes_b = torch.zeros((1, 1), device=hp.device)
                month_b = torch.zeros((1, 1), device=hp.device)
                diagt_b = torch.zeros((1, 1), device=hp.device)
                x_b[0, cols_list.index('en_nzdep_q')] = j - means['mean_nzdep']
                log_hr = net(x_b, codes_b, month_b, diagt_b).detach().cpu().numpy().squeeze()
            
            # Store
            log_hr_matrix[j-1, i] = log_hr
    
    # Compute HRs
    mean_hr = (log_hr_matrix.mean(axis=1))
    df = pd.DataFrame({'dep': range(1, 6), 'HR': mean_hr})
    df['diff_hr'] = np.exp(df['HR'].diff())
    print(df.describe())

    
if __name__ == '__main__':
    main()
