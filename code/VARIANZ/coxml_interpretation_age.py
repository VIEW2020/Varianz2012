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
    means = np.load(hp.data_pp_dir + 'means_' + hp.gender + '.npz')

    print('Add standard columns...')
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    num_cols = len(cols_list)

    #######################################################################################################
        
    print('Compute HRs...')

    # Trained models
    tmp = listdir(hp.log_dir + 'all/')
    models = ['all/' + i for i in tmp if '.pt' in i]    

    log_hr_matrix = np.zeros((len(range(30, 75)), len(models)))

    # Neural Net
    num_input = num_cols+1 if hp.nonprop_hazards else num_cols
    net = NetRNNFinal(num_input, num_embeddings+1, hp).to(hp.device) #+1 for zero padding
    net.eval()

    for i in range(len(models)):
        print('HRs for model {}'.format(i))
        
        # Restore variables from disk
        net.load_state_dict(torch.load(hp.log_dir + models[i], map_location=hp.device))

        # Compute risk for all ages
        for j in tqdm(range(30, 75)):
            with torch.no_grad():
                x_b = torch.zeros((1, num_cols), device=hp.device)
                codes_b = torch.zeros((1, 1), device=hp.device)
                month_b = torch.zeros((1, 1), device=hp.device)
                diagt_b = torch.zeros((1, 1), device=hp.device)
                x_b[0, cols_list.index('nhi_age')] = j - means['mean_age']
                risk_mod = net(x_b, codes_b, month_b, diagt_b).detach().cpu().numpy().squeeze() - risk_baseline
            
            # Store
            log_hr_matrix[j-30, i] = risk_mod
    
    # Compute HRs
    mean_hr = np.exp(log_hr_matrix.mean(axis=1))
    lCI, uCI = np.exp(sms.DescrStatsW(log_hr_matrix.transpose()).tconfint_mean())
    df = pd.DataFrame({'age': range(30, 75), 'HR': mean_hr, 'lCI': lCI, 'uCI': uCI})
    df['diff'] = df['HR'].diff()
    
    bp()
    
if __name__ == '__main__':
    main()
