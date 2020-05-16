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

from pdb import set_trace as bp


def main():
    _ = torch.manual_seed(hp.torch_seed)

    # Load data
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays.npz')
    
    x_trn = data['x_trn']
    time_trn = data['time_trn']
    event_trn = data['event_trn']
    codes_trn = data['codes_trn']
    month_trn = data['month_trn']
    diagt_trn = data['diagt_trn']
    
    x_tst = data['x_tst']
    time_tst = data['time_tst']
    event_tst = data['event_tst']
    codes_tst = data['codes_tst']
    month_tst = data['month_tst']
    diagt_tst = data['diagt_tst']
    
    if hp.nonprop_hazards:
        data = np.load(hp.data_pp_dir + 'data_arrays_nonprop_hazards.npz', allow_pickle=True)
        time_trn = data['time_trn_nonprop']
        with open(hp.data_pp_dir + 'labtrans.pkl', 'rb') as f:
            labtrans = pkl.load(f)

    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code.feather')

    rel_idx = [0,1,2,9,10,11,12,13,14,15] # exclude history covariates to avoid collinearity with individual codes
    x_trn = x_trn[:, rel_idx]
    x_tst = x_tst[:, rel_idx]

    ####################################################################################################### 

    # Neural Net
    n_inputs = x_trn.shape[1]+1 if hp.nonprop_hazards else x_trn.shape[1]
    net = NetAttention(n_inputs, df_index_code.shape[0]+1, hp) #+1 for zero padding

    # Trained models
    models = listdir(hp.log_dir)

    for i in range(len(models)):
        print('Model {}'.format(i))

        # Restore variables from disk
        net.load_state_dict(torch.load(hp.log_dir + models[i], map_location=hp.device))
   
        # Prediction
        print('Predicting survival...')
        model = CoxTime(net, labtrans=labtrans) if hp.nonprop_hazards else CoxCC(net)
        model.compute_baseline_hazards((torch.from_numpy(x_trn).flip(0),
                                        torch.from_numpy(codes_trn).flip(0), 
                                        torch.from_numpy(month_trn).flip(0),
                                        torch.from_numpy(diagt_trn).flip(0)),
                                       (torch.from_numpy(time_trn).flip(0), 
                                        torch.from_numpy(event_trn).flip(0)),
                                       sample=hp.sample_comp_bh, batch_size=hp.batch_size)
        surv = model.predict_surv_df((x_tst, codes_tst, month_tst, diagt_tst), batch_size=hp.batch_size)
        
        # Evaluation
        print('Evaluating...')
        ev = EvalSurv(surv, time_tst, event_tst, censor_surv='km')
        concordance = ev.concordance_td()
        print('Concordance: {:.6f}'.format(concordance))
        time_grid = np.linspace(time_tst.min(), time_tst.max(), 100)
        brier = ev.integrated_brier_score(time_grid)
        print('Brier score: {:.6f}'.format(brier))
        nbll = ev.integrated_nbll(time_grid)
        print('NBLL: {:.6f}'.format(nbll))
        
        # Log
        log(models[i], concordance, brier, nbll, hp) 

if __name__ == '__main__':
    main()
