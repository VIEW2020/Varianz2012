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
    
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code_' + hp.gender + '.feather')
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    print('Test on each fold...')
    for fold in range(hp.num_folds):
        print('Fold: {}'.format(fold))
        
        idx = (data['fold'] == fold)
        x = data['x'][idx]
        codes = data['codes'][idx]
        month = data['month'][idx]
        diagt = data['diagt'][idx]

        df = pd.DataFrame({'TIME': data['time'][idx], 'EVENT': data['event'][idx]})

        ####################################################################################################### 

        print('Create data loaders and tensors...')
        dataset = utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(codes), torch.from_numpy(month), torch.from_numpy(diagt))

        # Create batch queues
        loader = utils.DataLoader(dataset, batch_size = hp.batch_size, shuffle = False, drop_last = False)

        # Neural Net
        net = NetRNN(x.shape[1], df_index_code.shape[0]+1, hp).to(hp.device) #+1 for zero padding
        net.eval()

        # Trained models
        tmp = listdir(hp.log_dir + 'fold_' + str(fold) + '/')
        models = [i for i in tmp if '.pt' in i]
        base_surv_vec = np.zeros((1, len(models)))
        lph_matrix = np.zeros((x.shape[0], len(models)))
        risk_matrix = np.zeros((x.shape[0], len(models)))

        for i in range(len(models)):
            print('Model {}'.format(models[i]))
            # Restore variables from disk
            net.load_state_dict(torch.load(hp.log_dir + 'fold_' + str(fold) + '/' + models[i], map_location=hp.device))
    
            # Prediction
            log_partial_hazard = np.array([])
            print('Computing partial hazard for test data...')
            with torch.no_grad():
                for _, (x, codes, month, diagt) in enumerate(tqdm(loader)):
                    x, codes, month, diagt = x.to(hp.device), codes.to(hp.device), month.to(hp.device), diagt.to(hp.device)
                    log_partial_hazard = np.append(log_partial_hazard, net(x, codes, month, diagt).detach().cpu().numpy())
            lph_matrix[:, i] = log_partial_hazard
            base_surv_vec[0, i] = baseline_survival(df, log_partial_hazard).loc[1826]
            risk_matrix[:, i] = 100*(1-np.power(base_surv_vec[0, i], np.exp(log_partial_hazard)))

        df_base_surv = pd.DataFrame(base_surv_vec, columns=models)
        df_lph = pd.DataFrame(lph_matrix, columns=models)
        df_cml = pd.DataFrame(risk_matrix, columns=models)

        print('Ensemble...')
        lph_ensemble = lph_matrix.mean(axis=1)
        df_lph['ENSEMBLE'] = lph_ensemble
        base_surv_ensemble = baseline_survival(df, lph_ensemble).loc[1826]
        df_base_surv['ENSEMBLE'] = base_surv_ensemble        
        df_cml['ENSEMBLE'] = 100*(1-np.power(base_surv_ensemble, np.exp(lph_ensemble)))
        
        print('Saving...')
        df_base_surv.to_feather(hp.results_dir + 'df_base_surv_' + hp.gender + '_fold_' + str(fold) + '.feather')
        df_lph.to_feather(hp.results_dir + 'df_lph_' + hp.gender + '_fold_' + str(fold) + '.feather')
        df_cml.to_feather(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '.feather')


if __name__ == '__main__':
    main()
