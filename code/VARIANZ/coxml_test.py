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
from utils import *
from rnn_models import *
from hyperparameters import Hyperparameters

from os import listdir

from pdb import set_trace as bp


def main():
    hp = Hyperparameters()

    # Load data
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')

    x_trn = data['x_trn']
    codes_trn = data['codes_trn']
    month_trn = data['month_trn']
    diagt_trn = data['diagt_trn']
    
    x_tst = data['x_tst']
    codes_tst = data['codes_tst']
    month_tst = data['month_tst']
    diagt_tst = data['diagt_tst']
    
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code_' + hp.gender + '.feather')
    
    df_trn = pd.DataFrame(x_trn, columns=cols_list)
    df_trn['TIME'] = data['time_trn']
    df_trn['EVENT'] = data['event_trn']
    
    ####################################################################################################### 

    print('Create data loaders and tensors...')
    data_trn = utils.TensorDataset(torch.from_numpy(x_trn), torch.from_numpy(codes_trn), torch.from_numpy(month_trn), torch.from_numpy(diagt_trn))
    data_tst = utils.TensorDataset(torch.from_numpy(x_tst), torch.from_numpy(codes_tst), torch.from_numpy(month_tst), torch.from_numpy(diagt_tst))

    # Create batch queues
    trn_loader = utils.DataLoader(data_trn, batch_size = hp.batch_size, shuffle = False, drop_last = False)
    tst_loader = utils.DataLoader(data_tst, batch_size = hp.batch_size, shuffle = False, drop_last = False)

    # Neural Net
    net = NetRNN(x_trn.shape[1], df_index_code.shape[0]+1, hp).to(hp.device) #+1 for zero padding
    net.eval()

    # Trained models
    #tmp = listdir(hp.data_dir + 'log_' + hp.gender + '_iter3/')
    tmp = listdir(hp.log_dir)
    models = [i for i in tmp if '.pt' in i]
    risk_matrix = np.zeros((x_tst.shape[0], len(models)))

    for i in range(len(models)):
        print('Model {}'.format(models[i]))

        # Restore variables from disk
        net.load_state_dict(torch.load(hp.log_dir + models[i], map_location=hp.device))
   
        # Compute baseline survival using training data
        log_partial_hazard = np.array([])
        print('Computing baseline survival using training data...')
        with torch.no_grad():
            for _, (x_trn, codes_trn, month_trn, diagt_trn) in enumerate(tqdm(trn_loader)):
                x_trn, codes_trn, month_trn, diagt_trn = x_trn.to(hp.device), codes_trn.to(hp.device), month_trn.to(hp.device), diagt_trn.to(hp.device)
                log_partial_hazard = np.append(log_partial_hazard, net(x_trn, codes_trn, month_trn, diagt_trn).detach().cpu().numpy())
        base_surv = baseline_survival(df_trn, log_partial_hazard).loc[1826]
   
        # Prediction
        log_partial_hazard = np.array([])
        print('Computing partial hazard for test data...')
        with torch.no_grad():
            for _, (x_tst, codes_tst, month_tst, diagt_tst) in enumerate(tqdm(tst_loader)):
                x_tst, codes_tst, month_tst, diagt_tst = x_tst.to(hp.device), codes_tst.to(hp.device), month_tst.to(hp.device), diagt_tst.to(hp.device)
                log_partial_hazard = np.append(log_partial_hazard, net(x_tst, codes_tst, month_tst, diagt_tst).detach().cpu().numpy())
        
        print('Predicting...')
        risk_matrix[:, i] = 100*(1-np.power(base_surv, np.exp(log_partial_hazard)))
        
        print('Saving...')
        np.savez(hp.results_dir + 'risk_matrix_' + hp.gender + '.npz', risk_matrix=risk_matrix)
        save_obj(models, hp.results_dir + 'models.pkl')
        

if __name__ == '__main__':
    main()
