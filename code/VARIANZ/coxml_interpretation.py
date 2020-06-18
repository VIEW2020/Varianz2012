'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import sys
#sys.path.append('E:/Libraries/Python/')
#sys.path.append('..\\lib\\')
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

from pycox.evaluation import EvalSurv
from pycox.models import CoxCC, CoxTime

from deep_survival import *
from hyperparameters import Hyperparameters as hp

from os import listdir

import statsmodels.stats.api as sms
# import matplotlib.pyplot as plt

from pdb import set_trace as bp


def main():
    # Load data
    print('Load data...')
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code.feather')
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
    
    lookup = pd.concat([pharm_lookup, icd10_lookup], ignore_index=True, sort=False)
    
    print('Merge...')
    df_index_code['CODE'] = df_index_code['CODE'].astype(str)
    df_index_code = df_index_code.merge(lookup,   how='left', on=['CODE', 'TYPE'])

    #######################################################################################################
        
    print('Compute HRs...')
    
    # Trained models
    models = listdir(hp.log_dir)
    log_hr_matrix = np.zeros((df_index_code.shape[0], len(models)))

    # Neural Net
    n_inputs = 10
    net = NetAttention(n_inputs, df_index_code.shape[0]+1, hp) #+1 for zero padding

    for i in range(len(models)):
        print('HRs for model {}'.format(i))
        
        # Restore variables from disk
        net.load_state_dict(torch.load(hp.log_dir + models[i], map_location=hp.device))
        
        # HRs
        emb_weight = net.embed_codes.weight # primary diagnostic codes
        emb_weight = emb_weight[1:,:]
        fc_weight = net.fc.weight[:,10:].t()
        log_hr = torch.matmul(emb_weight, fc_weight).detach().cpu().numpy().squeeze()
        
        # Save
        log_hr_matrix[:, i] = log_hr
    
    # Compute HRs
    mean_hr = np.exp(log_hr_matrix.mean(axis=1))
    lCI, uCI = np.exp(sms.DescrStatsW(log_hr_matrix.transpose()).tconfint_mean())
    
    df_index_code['HR'] = mean_hr
    df_index_code['lCI'] = lCI
    df_index_code['uCI'] = uCI
    
    # Keep only codes existing as primary
    primary_codes = feather.read_dataframe(hp.data_pp_dir + 'primary_codes.feather')
    df_index_code = df_index_code[(df_index_code['TYPE'] == 0) | df_index_code['CODE'].isin(primary_codes['CLIN_CD_10'])]
    
    # Save
    df_index_code.sort_values(by=['TYPE', 'HR'], ascending=False, inplace=True)
    df_index_code.to_csv(hp.data_dir + 'hr.csv', index=False)    

if __name__ == '__main__':
    main()
