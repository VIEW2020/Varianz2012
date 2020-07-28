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

from deep_survival import *
from rnn_models import *
from utils import *
from hyperparameters import Hyperparameters

import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

from pdb import set_trace as bp


def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code_' + hp.gender + '.feather')

    print('Concat train/val/test data...')
    x = np.concatenate((data['x_trn'], data['x_val'], data['x_tst']))
    time = np.concatenate((data['time_trn'], data['time_val'], data['time_tst']))
    codes = np.concatenate((data['codes_trn'], data['codes_val'], data['codes_tst']))
    month = np.concatenate((data['month_trn'], data['month_val'], data['month_tst']))
    diagt = np.concatenate((data['diagt_trn'], data['diagt_val'], data['diagt_tst']))
    
    num_sample = 100000
    x = x[:num_sample]
    codes = codes[:num_sample]
    month = month[:num_sample]
    diagt = diagt[:num_sample]
    
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
    # Attributions
    data = np.load(hp.data_pp_dir + hp.attribution_alg + '_' + hp.gender + '.npz')
    x_attributions = data['x_attributions']
    codes_attributions = data['codes_attributions']
    
    # Storage
    log_hr_columns = np.zeros((num_cols, 3))
    log_hr_embeddings = np.zeros((num_embeddings, 3))    

    # Compute risk for masked standard columns
    # 0:'nhi_age', 1:'en_nzdep_q', 2:'hx_vdr_diabetes', 3:'hx_af', 4:'ph_bp_lowering_prior_6mths', 5:'ph_lipid_lowering_prior_6mths', 6:'ph_antiplat_anticoag_prior_6mths', 
    # 7:'age_X_bp', 8:'age_X_diabetes', 9:'age_X_af', 
    # 10:'bp_X_diabetes', 11:'antiplat_anticoag_X_diabetes', 12:'bp_X_af', 
    # 13:'en_prtsd_eth_2', 14:'en_prtsd_eth_3', 15:'en_prtsd_eth_9', 16:'en_prtsd_eth_43'    
    print('HRs for pre-specified predictors...')
    for i in tqdm(range(num_cols)):
        if cols_list[i] in ['nhi_age', 'en_nzdep_q', 'age_X_bp', 'age_X_diabetes', 'age_X_af']:
            x_attributions[:, i] = np.divide(x_attributions[:, i], x[:, i], out=np.zeros_like(x_attributions[:, i]), where=x[:, i]!=0)
        vec = x_attributions[x[:, i]!=0, i]
        # Store
        log_hr_columns[i, 0] = vec.mean()
        log_hr_columns[i, 1] = np.quantile(vec, 0.025)
        log_hr_columns[i, 2] = np.quantile(vec, 0.975)

    # Compute risk for masked embeddings
    print('HRs for embedding...')
    for i in tqdm(range(num_embeddings)):
        vec = np.ma.array(codes_attributions, mask=(codes!=(i+1))).compressed()
        # Store
        log_hr_embeddings[i, 0] = vec.mean()
        log_hr_embeddings[i, 1] = np.quantile(vec, 0.025)
        log_hr_embeddings[i, 2] = np.quantile(vec, 0.975)
    
    # Compute HRs
    log_hr = np.concatenate((log_hr_columns, log_hr_embeddings))
    df_index_code['HR'] = np.exp(log_hr[:, 0])
    df_index_code['lCI'] = np.exp(log_hr[:, 1])
    df_index_code['uCI'] = np.exp(log_hr[:, 2])
        
    # Save
    df_index_code.sort_values(by=['TYPE', 'lCI'], ascending=False, inplace=True)
    df_index_code.to_csv(hp.data_dir + 'hr_' + hp.attribution_alg + '_' + hp.gender + '.csv', index=False)
    
    # Keep only codes existing as primary
    primary_codes = feather.read_dataframe(hp.data_pp_dir + 'primary_codes.feather')
    df_index_code = df_index_code[(df_index_code['TYPE'] == 0) | (df_index_code['TYPE'] == 2) | df_index_code['CODE'].isin(primary_codes['CLIN_CD_10'])]    

    # Save
    df_index_code.to_csv(hp.data_dir + 'hr_reduced_' + hp.attribution_alg + '_' + hp.gender + '.csv', index=False)

if __name__ == '__main__':
    main()
