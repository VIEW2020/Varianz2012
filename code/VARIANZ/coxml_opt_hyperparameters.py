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
from hyperparameters import Hyperparameters
import optuna

from pdb import set_trace as bp


def objective(trial, data, df_index_code):
    hp = Hyperparameters(trial)
    #hp = Hyperparameters()
    print(trial.params)

    idx_trn = (data['fold'] != 99)
    x_trn = data['x'][idx_trn]
    time_trn = data['time'][idx_trn]
    event_trn = data['event'][idx_trn]
    codes_trn = data['codes'][idx_trn]
    month_trn = data['month'][idx_trn]
    diagt_trn = data['diagt'][idx_trn]

    idx_val = (data['fold'] == 99)
    x_val = data['x'][idx_val]
    time_val = data['time'][idx_val]
    event_val = data['event'][idx_val]
    codes_val = data['codes'][idx_val]
    month_val = data['month'][idx_val]
    diagt_val = data['diagt'][idx_val]

    # could move this outside objective function for efficiency
    sort_idx_trn, case_idx_trn, max_idx_control_trn = sort_and_case_indices(x_trn, time_trn, event_trn)
    sort_idx_val, case_idx_val, max_idx_control_val = sort_and_case_indices(x_val, time_val, event_val)

    x_trn, time_trn, event_trn = x_trn[sort_idx_trn], time_trn[sort_idx_trn], event_trn[sort_idx_trn]
    codes_trn, month_trn, diagt_trn = codes_trn[sort_idx_trn], month_trn[sort_idx_trn], diagt_trn[sort_idx_trn]

    x_val, time_val, event_val = x_val[sort_idx_val], time_val[sort_idx_val], event_val[sort_idx_val]
    codes_val, month_val, diagt_val = codes_val[sort_idx_val], month_val[sort_idx_val], diagt_val[sort_idx_val]
    
    ####################################################################################################### 

    print('Create data loaders and tensors...')
    case_trn = utils.TensorDataset(torch.from_numpy(x_trn[case_idx_trn]),
                                   torch.from_numpy(time_trn[case_idx_trn]),
                                   torch.from_numpy(max_idx_control_trn),
                                   torch.from_numpy(codes_trn[case_idx_trn]),
                                   torch.from_numpy(month_trn[case_idx_trn]),
                                   torch.from_numpy(diagt_trn[case_idx_trn]))
    case_val = utils.TensorDataset(torch.from_numpy(x_val[case_idx_val]),
                                   torch.from_numpy(time_val[case_idx_val]),
                                   torch.from_numpy(max_idx_control_val),
                                   torch.from_numpy(codes_val[case_idx_val]),
                                   torch.from_numpy(month_val[case_idx_val]),
                                   torch.from_numpy(diagt_val[case_idx_val]))

    x_trn, x_val = torch.from_numpy(x_trn), torch.from_numpy(x_val)
    time_trn, time_val = torch.from_numpy(time_trn), torch.from_numpy(time_val)
    event_trn, event_val = torch.from_numpy(event_trn), torch.from_numpy(event_val)
    codes_trn, codes_val = torch.from_numpy(codes_trn), torch.from_numpy(codes_val)
    month_trn, month_val = torch.from_numpy(month_trn), torch.from_numpy(month_val)
    diagt_trn, diagt_val = torch.from_numpy(diagt_trn), torch.from_numpy(diagt_val)

    # Create batch queues
    trn_loader = utils.DataLoader(case_trn, batch_size = hp.batch_size, shuffle = True,  drop_last = True)
    val_loader = utils.DataLoader(case_val, batch_size = hp.batch_size, shuffle = False, drop_last = False)

    print('Train...')
    # Neural Net
    hp.model_name = str(trial.number) + '_' + hp.model_name 
    num_input = x_trn.shape[1]+1 if hp.nonprop_hazards else x_trn.shape[1]
    net = NetRNNFinal(num_input, df_index_code.shape[0]+1, hp).to(hp.device) #+1 for zero padding
    criterion = CoxPHLoss().to(hp.device)
    optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)
    
    best, num_bad_epochs = 100., 0
    for epoch in range(1000):
        trn(trn_loader, x_trn, codes_trn, month_trn, diagt_trn, net, criterion, optimizer, hp)
        loss_val = val(val_loader, x_val, codes_val, month_val, diagt_val, net, criterion, epoch, hp)
        # early stopping
        if loss_val < best:
            print('############### Saving good model ###############################')
            torch.save(net.state_dict(), hp.log_dir + hp.model_name)
            best = loss_val
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
            if num_bad_epochs == hp.patience:
                break
        # pruning
        trial.report(best, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    print('Done')
    return best


def main():
    pp = Hyperparameters()
    
    print('Load data...')
    data = np.load(pp.data_pp_dir + 'data_arrays_' + pp.gender + '.npz')
    df_index_code = feather.read_dataframe(pp.data_pp_dir + 'df_index_code_' + pp.gender + '.feather')
    
    print('Begin study...')
    #study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner())
    study = optuna.create_study(sampler=optuna.samplers.GridSampler({'summarize': ['output_attention']}), pruner=optuna.pruners.NopPruner())
    study.optimize(lambda trial: objective(trial, data, df_index_code), n_trials=1)
    
    print('Save...')
    save_obj(study, pp.log_dir + 'study_' + pp.gender + '.pkl')


if __name__ == '__main__':
    main()
