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
from hyperparameters import Hyperparameters

from os import listdir

from pdb import set_trace as bp


def main():
    hp = Hyperparameters()
    study_males = load_obj(hp.data_dir + 'log_males_iter0/study_males.pkl')
    print(study_males.best_params)
    print(study_males.best_value)
    study_females = load_obj(hp.data_dir + 'log_females_iter0/study_females.pkl')
    print(study_females.best_params)
    print(study_females.best_value)
    study_males = load_obj(hp.data_dir + 'log_males_iter1/study_males.pkl')
    print(study_males.best_params)
    print(study_males.best_value)    
    study_females = load_obj(hp.data_dir + 'log_females_iter1/study_females.pkl')
    print(study_females.best_params)
    print(study_females.best_value)    
    study_males = load_obj(hp.data_dir + 'log_males_iter2/study_males.pkl')
    print(study_males.best_params)
    print(study_males.best_value)    
    study_females = load_obj(hp.data_dir + 'log_females_iter2/study_females.pkl')
    print(study_females.best_params)
    print(study_females.best_value)
    bp()

    return

    # Load data
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    
    x_tst = data['x_tst']
    time_tst = data['time_tst']
    event_tst = data['event_tst']
    codes_tst = data['codes_tst']
    month_tst = data['month_tst']
    diagt_tst = data['diagt_tst']
    
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code_' + hp.gender + '.feather')
    
    df_tst = pd.DataFrame(x_tst, columns=cols_list)
    df_tst['TIME'] = time_tst
    df_tst['EVENT'] = event_tst 

    ####################################################################################################### 

    print('Create data loaders and tensors...')
    data_tst = utils.TensorDataset(torch.from_numpy(x_tst),
                                   torch.from_numpy(codes_tst),
                                   torch.from_numpy(month_tst),
                                   torch.from_numpy(diagt_tst))

    # Create batch queues
    tst_loader = utils.DataLoader(data_tst, batch_size = hp.batch_size, shuffle = False, drop_last = False)

    # Neural Net
    net = NetAttention(x_tst.shape[1], df_index_code.shape[0]+1, hp).to(hp.device) #+1 for zero padding

    # Trained models
    models = listdir(hp.data_dir + 'log_' + hp.gender + '/')

    for i in range(len(models)):
        print('Model {}'.format(i))

        # Restore variables from disk
        net.load_state_dict(torch.load(hp.data_dir + 'log_' + hp.gender + '/' + models[i], map_location=hp.device))
   
        # Prediction
        log_partial_hazard = np.array([])
        print('Computing partial hazard...')
        net.eval()
        with torch.no_grad():
            for _, (x_tst, codes_tst, month_tst, diagt_tst) in enumerate(tqdm(tst_loader)):
                x_tst, codes_tst, month_tst, diagt_tst = x_tst.to(hp.device), codes_tst.to(hp.device), month_tst.to(hp.device), diagt_tst.to(hp.device)
                log_partial_hazard = np.append(log_partial_hazard, net(x_tst, codes_tst, month_tst, diagt_tst).detach().cpu().numpy())
        
        print('Predicting...')
        base_surv = baseline_survival(df_tst, log_partial_hazard).loc[1826]
        df_tst['RISK_' + str(i)] = 100*(1-np.power(base_surv, np.exp(log_partial_hazard)))

    df_tst['RISK'] = df_tst.filter(like='RISK').mean(axis=1)
    df_tst.to_feather(hp.plot_dir + 'df_tst_cml_' + hp.gender + '.feather')

    ## Prediction
    #print('Predicting survival...')
    #model = CoxTime(net, labtrans=labtrans) if hp.nonprop_hazards else CoxCC(net)
    #model.compute_baseline_hazards((torch.from_numpy(x_trn).flip(0),
                                    #torch.from_numpy(codes_trn).flip(0), 
                                    #torch.from_numpy(month_trn).flip(0),
                                    #torch.from_numpy(diagt_trn).flip(0)),
                                    #(torch.from_numpy(time_trn).flip(0), 
                                    #torch.from_numpy(event_trn).flip(0)),
                                    #sample=hp.sample_comp_bh, batch_size=hp.batch_size)
    #surv = model.predict_surv_df((x_tst, codes_tst, month_tst, diagt_tst), batch_size=hp.batch_size)
    
    ## Evaluation
    #print('Evaluating...')
    #ev = EvalSurv(surv, time_tst, event_tst, censor_surv='km')
    #concordance = ev.concordance_td()
    #print('Concordance: {:.6f}'.format(concordance))
    #time_grid = np.linspace(time_tst.min(), time_tst.max(), 100)
    #brier = ev.integrated_brier_score(time_grid)
    #print('Brier score: {:.6f}'.format(brier))
    #nbll = ev.integrated_nbll(time_grid)
    #print('NBLL: {:.6f}'.format(nbll))
    
    ## Log
    #log(models[i], concordance, brier, nbll, hp) 

if __name__ == '__main__':
    main()
