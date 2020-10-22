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
from utils import *
from rnn_models import *
from EvalSurv import EvalSurv
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
        for swap in range(2):
            print('Fold: {} Swap: {}'.format(fold, swap))
            
            idx = (data['fold'][:, fold] == swap)
            x = data['x'][idx]
            codes = data['codes'][idx]
            month = data['month'][idx]
            diagt = data['diagt'][idx]            

            ####################################################################################################### 

            print('Create data loaders and tensors...')
            dataset = utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(codes), torch.from_numpy(month), torch.from_numpy(diagt))

            # Create batch queues
            loader = utils.DataLoader(dataset, batch_size = hp.batch_size, shuffle = False, drop_last = False)

            # Neural Net
            net = NetRNNFinal(x.shape[1], df_index_code.shape[0]+1, hp).to(hp.device) #+1 for zero padding
            net.eval()

            # Trained models
            tmp = listdir(hp.log_dir + 'fold_' + str(fold) + '_' + str(1-swap) + '/')
            models = ['fold_' + str(fold) + '_' + str(1-swap) + '/' + i for i in tmp if '.pt' in i]
            lph_matrix = np.zeros((x.shape[0], len(models)))

            for i in range(len(models)):
                print('Model {}'.format(models[i]))
                # Restore variables from disk
                net.load_state_dict(torch.load(hp.log_dir + models[i], map_location=hp.device))
        
                # Prediction
                log_partial_hazard = np.array([])
                print('Computing partial hazard for test data...')
                with torch.no_grad():
                    for _, (x, codes, month, diagt) in enumerate(tqdm(loader)):
                        x, codes, month, diagt = x.to(hp.device), codes.to(hp.device), month.to(hp.device), diagt.to(hp.device)
                        log_partial_hazard = np.append(log_partial_hazard, net(x, codes, month, diagt).detach().cpu().numpy())
                lph_matrix[:, i] = log_partial_hazard

            print('Create dataframe...')
            df_lph = pd.DataFrame(lph_matrix, columns=models)
            
            print('Saving log proportional hazards for fold...')
            df_lph.to_feather(hp.results_dir + 'df_lph_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '.feather')
    
    
if __name__ == '__main__':
    main()
