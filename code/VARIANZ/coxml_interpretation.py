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

import statsmodels.stats.api as sms
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
    
    print('Merge with lookup table...')
    lookup = pd.concat([pharm_lookup, icd10_lookup], ignore_index=True, sort=False)
    df_index_code['CODE'] = df_index_code['CODE'].astype(str)
    df_index_code = df_index_code.merge(lookup,   how='left', on=['CODE', 'TYPE'])
    num_embeddings = df_index_code.shape[0]
    
    print('Concat train/val/test data...')
    x = np.concatenate((data['x_trn'], data['x_val'], data['x_tst']))
    time = np.concatenate((data['time_trn'], data['time_val'], data['time_tst']))
    codes = np.concatenate((data['codes_trn'], data['codes_val'], data['codes_tst']))
    month = np.concatenate((data['month_trn'], data['month_val'], data['month_tst']))
    diagt = np.concatenate((data['diagt_trn'], data['diagt_val'], data['diagt_tst']))
    
    print('Create data loader...')
    data_tensors = utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(time), torch.from_numpy(codes), torch.from_numpy(month), torch.from_numpy(diagt))
    data_loader = utils.DataLoader(data_tensors, batch_size = hp.batch_size, shuffle = False,  drop_last = False)

    print('Add standard columns...')
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    num_cols = len(cols_list)
    df_cols = pd.DataFrame({'TYPE': 2, 'DESCRIPTION': cols_list})
    df_index_code = pd.concat([df_cols, df_index_code], sort=False)

    #######################################################################################################
        
    print('Compute HRs...')
    
    # Trained model
    log_hr_columns = np.zeros((num_cols, 3))
    log_hr_embeddings = np.zeros((num_embeddings, 3))

    # Neural Net
    num_input = x.shape[1]+1 if hp.nonprop_hazards else x.shape[1]
    net = NetRNN(num_input, num_embeddings+1, hp).to(hp.device) #+1 for zero padding
    net.load_state_dict(torch.load(hp.log_dir + hp.test_model, map_location=hp.device))
    
    # Compute risk for everyone
    risk = np.zeros(x.shape[0])
    net.eval()
    with torch.no_grad():
        for batch_idx, (x_b, time_b, code_b, month_b, diagt_b) in enumerate(tqdm(data_loader)):
            x_b = x_b.to(hp.device)
            time_b = time_b.to(hp.device)
            code_b = code_b.to(hp.device)
            month_b = month_b.to(hp.device)
            diagt_b = diagt_b.to(hp.device)
            risk[(batch_idx*hp.batch_size):min((batch_idx+1)*hp.batch_size, risk.shape[0])] = net(x_b, code_b, month_b, diagt_b, time_b).detach().cpu().numpy().squeeze()

    # Compute risk for masked standard columns
    # 0:'nhi_age', 1:'en_nzdep_q', 2:'hx_vdr_diabetes', 3:'hx_af', 4:'ph_bp_lowering_prior_6mths', 5:'ph_lipid_lowering_prior_6mths', 6:'ph_antiplat_anticoag_prior_6mths', 
    # 7:'age_X_bp', 8:'age_X_diabetes', 9:'age_X_af', 
    # 10:'bp_X_diabetes', 11:'antiplat_anticoag_X_diabetes', 12:'bp_X_af', 
    # 13:'en_prtsd_eth_2', 14:'en_prtsd_eth_3', 15:'en_prtsd_eth_9', 16:'en_prtsd_eth_43'    
    for i in tqdm(range(num_cols)):
        if cols_list[i] == 'nhi_age':
            min_age = min(x[:,i])
            idx = (x[:,i] >= (min_age+1))
            x_red, time_red, codes_red, month_red, diagt_red, risk_red = x[idx], time[idx], codes[idx], month[idx], diagt[idx], risk[idx]
            x_red[:,i] = x_red[:,i]-1
            x_red[:,7:10] = x_red[:,7:10]-1
            x_red[:,7] = x_red[:,7]*x_red[:,4]
            x_red[:,8] = x_red[:,8]*x_red[:,2]
            x_red[:,9] = x_red[:,9]*x_red[:,3]
        elif cols_list[i] == 'en_nzdep_q':
            min_nzdep = min(x[:,i])
            idx = (x[:,i] >= (min_nzdep+1))
            x_red, time_red, codes_red, month_red, diagt_red, risk_red = x[idx], time[idx], codes[idx], month[idx], diagt[idx], risk[idx]
            x_red[:,i] = x_red[:,i]-1
        elif cols_list[i] == 'hx_vdr_diabetes':
            idx = x[:,i].astype(bool)
            x_red, time_red, codes_red, month_red, diagt_red, risk_red = x[idx], time[idx], codes[idx], month[idx], diagt[idx], risk[idx]
            x_red[:,i] = 0
            x_red[:,8] = 0
            x_red[:,10] = 0
            x_red[:,11] = 0
        elif cols_list[i] == 'hx_af':
            idx = x[:,i].astype(bool)
            x_red, time_red, codes_red, month_red, diagt_red, risk_red = x[idx], time[idx], codes[idx], month[idx], diagt[idx], risk[idx]
            x_red[:,i] = 0
            x_red[:,9] = 0
            x_red[:,12] = 0
        elif cols_list[i] == 'ph_bp_lowering_prior_6mths':
            idx = x[:,i].astype(bool)
            x_red, time_red, codes_red, month_red, diagt_red, risk_red = x[idx], time[idx], codes[idx], month[idx], diagt[idx], risk[idx]
            x_red[:,i] = 0
            x_red[:,7] = 0
            x_red[:,10] = 0
            x_red[:,12] = 0
        elif cols_list[i] == 'ph_antiplat_anticoag_prior_6mths':
            idx = x[:,i].astype(bool)
            x_red, time_red, codes_red, month_red, diagt_red, risk_red = x[idx], time[idx], codes[idx], month[idx], diagt[idx], risk[idx]
            x_red[:,i] = 0
            x_red[:,11] = 0
        elif cols_list[i] in ['ph_lipid_lowering_prior_6mths', 'en_prtsd_eth_2', 'en_prtsd_eth_3', 'en_prtsd_eth_9', 'en_prtsd_eth_43']:
            idx = x[:,i].astype(bool)
            x_red, time_red, codes_red, month_red, diagt_red, risk_red = x[idx], time[idx], codes[idx], month[idx], diagt[idx], risk[idx]
            x_red[:,i] = 0
        else:
            continue
        data_tensors_red = utils.TensorDataset(torch.from_numpy(x_red), torch.from_numpy(time_red), torch.from_numpy(codes_red), torch.from_numpy(month_red), torch.from_numpy(diagt_red))
        data_loader_red = utils.DataLoader(data_tensors_red, batch_size = hp.batch_size, shuffle = False,  drop_last = False)
        risk_masked = np.zeros_like(risk_red)
        with torch.no_grad():
            for batch_idx, (x_b, time_b, code_b, month_b, diagt_b) in enumerate(tqdm(data_loader_red)):
                x_b = x_b.to(hp.device)
                time_b = time_b.to(hp.device)
                code_b = code_b.to(hp.device)
                month_b = month_b.to(hp.device)
                diagt_b = diagt_b.to(hp.device)
                risk_masked[(batch_idx*hp.batch_size):min((batch_idx+1)*hp.batch_size, risk_masked.shape[0])] = net(x_b, code_b, month_b, diagt_b, time_b).detach().cpu().numpy().squeeze()
        diff = risk_red - risk_masked
        
        # Store
        log_hr_columns[i, 0] = diff.mean()
        log_hr_columns[i, 1] = np.quantile(diff, 0.025)
        log_hr_columns[i, 2] = np.quantile(diff, 0.975)

    # Compute risk for masked embeddings
    for i in tqdm(range(num_embeddings)):
        print('HRs for embedding {}'.format(i))
        mask = (codes==(i+1))
        idx = mask.max(axis=1)
        x_red, time_red, codes_red, month_red, diagt_red, risk_red, mask_red = x[idx], time[idx], codes[idx], month[idx], diagt[idx], risk[idx], mask[idx]
        data_tensors_red = utils.TensorDataset(torch.from_numpy(x_red), torch.from_numpy(time_red), torch.from_numpy(codes_red), torch.from_numpy(month_red), torch.from_numpy(diagt_red), torch.from_numpy(mask_red))
        data_loader_red = utils.DataLoader(data_tensors_red, batch_size = hp.batch_size, shuffle = False,  drop_last = False)
        risk_masked = np.zeros_like(risk_red)
        with torch.no_grad():
            for batch_idx, (x_b, time_b, code_b, month_b, diagt_b, mask_b) in enumerate(tqdm(data_loader_red)):
                x_b = x_b.to(hp.device)
                time_b = time_b.to(hp.device)
                code_b = code_b.to(hp.device)
                month_b = month_b.to(hp.device)
                diagt_b = diagt_b.to(hp.device)
                mask_b = mask_b.to(hp.device)
                risk_masked[(batch_idx*hp.batch_size):min((batch_idx+1)*hp.batch_size, risk_masked.shape[0])] = net(x_b, code_b, month_b, diagt_b, time_b, mask_b).detach().cpu().numpy().squeeze()
        diff = risk_red - risk_masked
        
        # Store
        log_hr_embeddings[i, 0] = diff.mean()
        log_hr_embeddings[i, 1] = np.quantile(diff, 0.025)
        log_hr_embeddings[i, 2] = np.quantile(diff, 0.975)
    
    # Compute HRs
    log_hr = np.concatenate((log_hr_columns, log_hr_embeddings))
    df_index_code['HR'] = np.exp(log_hr[:, 0])
    df_index_code['lCI'] = np.exp(log_hr[:, 1])
    df_index_code['uCI'] = np.exp(log_hr[:, 2])
        
    # Save
    df_index_code.sort_values(by=['TYPE', 'lCI'], ascending=False, inplace=True)
    df_index_code.to_csv(hp.data_dir + 'hr_' + hp.gender + '.csv', index=False)
    
    # Keep only codes existing as primary
    primary_codes = feather.read_dataframe(hp.data_pp_dir + 'primary_codes.feather')
    df_index_code = df_index_code[(df_index_code['TYPE'] == 0) | (df_index_code['TYPE'] == 2) | df_index_code['CODE'].isin(primary_codes['CLIN_CD_10'])]    

    # Save
    df_index_code.to_csv(hp.data_dir + 'hr_reduced_' + hp.gender + '.csv', index=False)

if __name__ == '__main__':
    main()
