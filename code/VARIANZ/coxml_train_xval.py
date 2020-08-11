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

import torch
import torch.utils.data as utils
import torch.optim as optim
import torch.nn.functional as F

from deep_survival import *
from utils import *
from rnn_models import *
from EvalSurv import EvalSurv
from hyperparameters import Hyperparameters

from datetime import datetime
from pdb import set_trace as bp


def main():
    hp = Hyperparameters()
    
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code_' + hp.gender + '.feather')
    
    print('Train on each fold...')
    for fold in range(hp.num_folds):
        print('Fold: {}'.format(fold))
        
        idx = (data['fold'] != fold) & (data['fold'] != 99)
        x = data['x'][idx]
        time = data['time'][idx]
        event = data['event'][idx]
        codes = data['codes'][idx]
        month = data['month'][idx]
        diagt = data['diagt'][idx]

        sort_idx, case_idx, max_idx_control = sort_and_case_indices(x, time, event)
        x, time, event = x[sort_idx], time[sort_idx], event[sort_idx]
        codes, month, diagt = codes[sort_idx], month[sort_idx], diagt[sort_idx]
        
        print('Create data loaders and tensors...')
        case = utils.TensorDataset(torch.from_numpy(x[case_idx]),
                                   torch.from_numpy(time[case_idx]),
                                   torch.from_numpy(max_idx_control),
                                   torch.from_numpy(codes[case_idx]),
                                   torch.from_numpy(month[case_idx]),
                                   torch.from_numpy(diagt[case_idx]))

        x = torch.from_numpy(x)
        time = torch.from_numpy(time)
        event = torch.from_numpy(event)
        codes = torch.from_numpy(codes)
        month = torch.from_numpy(month)
        diagt = torch.from_numpy(diagt)

        for trial in range(hp.num_trials):
            print('Trial: {}'.format(trial))
            
            # Create batch queues
            trn_loader = utils.DataLoader(case, batch_size = hp.batch_size, shuffle = True,  drop_last = True)

            print('Train...')
            # Neural Net
            hp.model_name = str(trial) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.pt'
            num_input = x.shape[1]+1 if hp.nonprop_hazards else x.shape[1]
            net = NetRNN(num_input, df_index_code.shape[0]+1, hp).to(hp.device) #+1 for zero padding
            criterion = CoxPHLoss().to(hp.device)
            optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)
            
            for epoch in range(hp.max_epochs):
                trn(trn_loader, x, codes, month, diagt, net, criterion, optimizer, hp)
            torch.save(net.state_dict(), hp.log_dir + 'fold_' + str(fold) + '/' + hp.model_name)
            print('Done')        
            
            
if __name__ == '__main__':
    main()
