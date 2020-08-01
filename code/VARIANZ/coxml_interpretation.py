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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

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
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code_' + hp.gender + '.feather')
    
    print('Create list of codes...')
    pharm_lookup = feather.read_dataframe(hp.data_dir + 'CURRENT_VIEW_PHARMS_LOOKUP.feather')
    icd10_lookup = feather.read_dataframe(hp.data_dir + 'CURRENT_ICD10_ALL_LOOKUP.feather')

    pharm_lookup = pharm_lookup[['CHEMICAL_ID', 'CHEMICAL_NAME']]
    pharm_lookup.rename(columns={'CHEMICAL_ID': 'CODE', 'CHEMICAL_NAME': 'DESCRIPTION'}, inplace=True)
    pharm_lookup['CODE'] = pharm_lookup['CODE'].fillna(0).astype(int).astype(str)
    pharm_lookup.drop_duplicates(subset='CODE', inplace=True)
    pharm_lookup['TYPE'] = 0
    
    icd10_lookup = icd10_lookup[['code', 'code_description']]
    icd10_lookup.rename(columns={'code': 'CODE', 'code_description': 'DESCRIPTION'}, inplace=True)
    icd10_lookup['CODE'] = icd10_lookup['CODE'].astype(str)
    icd10_lookup.drop_duplicates(subset='CODE', inplace=True)
    icd10_lookup['TYPE'] = 1

    print('Determine most frequent code type...')
    pharm_lookup['DIAG_TYPE'] = 0
    most_freq_diagt = feather.read_dataframe(hp.data_pp_dir + 'most_freq_diagt_' + hp.gender + '.feather')
    most_freq_diagt.rename(columns={'CLIN_CD_10': 'CODE'}, inplace=True)
    icd10_lookup = icd10_lookup.merge(most_freq_diagt,   how='left', on='CODE')
    
    print('Merge with lookup table...')
    lookup = pd.concat([pharm_lookup, icd10_lookup], ignore_index=True, sort=False)
    df_index_code['CODE'] = df_index_code['CODE'].astype(str)
    df_index_code = df_index_code.merge(lookup,   how='left', on=['CODE', 'TYPE'])
    num_embeddings = df_index_code.shape[0]
    
    print('Add standard columns...')
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    num_cols = len(cols_list)
    df_cols = pd.DataFrame({'TYPE': 2, 'DESCRIPTION': cols_list})
    df_index_code = pd.concat([df_cols, df_index_code], sort=False)

    #######################################################################################################
        
    print('Compute HRs...')

    # Trained models
    tmp = listdir(hp.data_dir + 'log_' + hp.gender + '_iter3/')
    models = [i for i in tmp if '.pt' in i]
    log_hr_columns = np.zeros((num_cols, len(models)))
    log_hr_embeddings = np.zeros((num_embeddings, len(models)))

    # Neural Net
    num_input = num_cols+1 if hp.nonprop_hazards else num_cols
    net = NetRNN(num_input, num_embeddings+1, hp).to(hp.device) #+1 for zero padding
    net.eval()

    for i in range(len(models)):
        print('HRs for model {}'.format(i))
        
        # Restore variables from disk
        net.load_state_dict(torch.load(hp.data_dir + 'log_' + hp.gender + '_iter3/' + models[i], map_location=hp.device))

        with torch.no_grad():
            x_b = torch.zeros((1, num_cols), device=hp.device)
            codes_b = torch.zeros((1, 1), device=hp.device)
            month_b = torch.zeros((1, 1), device=hp.device)
            diagt_b = torch.zeros((1, 1), device=hp.device)
            risk_baseline = net(x_b, codes_b, month_b, diagt_b).detach().cpu().numpy().squeeze()

        # Compute risk for masked standard columns
        for j in tqdm(range(num_cols)):
            with torch.no_grad():
                x_b = torch.zeros((1, num_cols), device=hp.device)
                codes_b = torch.zeros((1, 1), device=hp.device)
                month_b = torch.zeros((1, 1), device=hp.device)
                diagt_b = torch.zeros((1, 1), device=hp.device)
                x_b[0, j] = 1
                risk_mod = net(x_b, codes_b, month_b, diagt_b).detach().cpu().numpy().squeeze() - risk_baseline
            
            # Store
            log_hr_columns[j, i] = risk_mod

        # Compute risk for masked embeddings
        for j in tqdm(range(num_embeddings)):
            with torch.no_grad():
                x_b = torch.zeros((1, num_cols), device=hp.device)
                codes_b = torch.zeros((1, 1), device=hp.device)
                month_b = torch.zeros((1, 1), device=hp.device)
                diagt_b = torch.zeros((1, 1), device=hp.device)
                codes_b[0] = (j+1)
                month_b[0] = 59
                diagt_b[0] = df_index_code['DIAG_TYPE'].values[j]
                risk_mod = net(x_b, codes_b, month_b, diagt_b).detach().cpu().numpy().squeeze() - risk_baseline
            
            # Store
            log_hr_embeddings[j, i] = risk_mod
    
    # Compute HRs
    log_hr_matrix = np.concatenate((log_hr_columns, log_hr_embeddings))
    mean_hr = np.exp(log_hr_matrix.mean(axis=1))
    lCI, uCI = np.exp(sms.DescrStatsW(log_hr_matrix.transpose()).tconfint_mean())
    df_index_code['HR'] = mean_hr
    df_index_code['lCI'] = lCI
    df_index_code['uCI'] = uCI
        
    # Save
    df_index_code.sort_values(by=['TYPE', 'HR'], ascending=False, inplace=True)
    df_index_code.to_csv(hp.data_dir + 'hr_addcodes_' + hp.gender + '.csv', index=False)
    
    # # Keep only codes existing as primary
    # primary_codes = feather.read_dataframe(hp.data_pp_dir + 'primary_codes.feather')
    # df_index_code = df_index_code[(df_index_code['TYPE'] == 0) | (df_index_code['TYPE'] == 2) | df_index_code['CODE'].isin(primary_codes['CLIN_CD_10'])]    

    # # Save
    # df_index_code.to_csv(hp.data_dir + 'hr_addcodes_reduced_' + hp.gender + '.csv', index=False)

if __name__ == '__main__':
    main()
