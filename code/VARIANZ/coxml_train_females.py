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

from deep_survival import *
from hyperparameters import Hyperparameters as hp

from pdb import set_trace as bp


def main():
    #_ = torch.manual_seed(hp.torch_seed)

    # Load data
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays_females.npz')
    
    x_trn = data['x_trn']
    time_trn = data['time_trn']
    event_trn = data['event_trn']
    codes_trn = data['codes_trn']
    month_trn = data['month_trn']
    diagt_trn = data['diagt_trn']
    case_idx_trn = data['case_idx_trn']
    max_idx_control_trn = data['max_idx_control_trn']
    
    x_val = data['x_val']
    time_val = data['time_val']
    event_val = data['event_val']
    codes_val = data['codes_val']
    month_val = data['month_val']
    diagt_val = data['diagt_val']
    case_idx_val = data['case_idx_val']
    max_idx_control_val = data['max_idx_control_val']
    
    df_index_code = feather.read_dataframe(hp.data_pp_dir + 'df_index_code_females.feather')
    
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
    n_inputs = x_trn.shape[1]+1 if hp.nonprop_hazards else x_trn.shape[1]
    net = NetAttention(n_inputs, df_index_code.shape[0]+1, hp).to(hp.device) #+1 for zero padding
    criterion = CoxPHLoss().to(hp.device)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    
    best, num_bad_epochs = 1e10, 0
    for epoch in range(hp.max_epochs):
        # print(epoch)
        trn(trn_loader, x_trn, codes_trn, month_trn, diagt_trn, net, criterion, optimizer, hp)
        loss_val = val(val_loader, x_val, codes_val, month_val, diagt_val, net, criterion, epoch, hp)
        # early stopping
        if loss_val < best:
            print('############### Saving good model ###############################')
            torch.save(net.state_dict(), hp.data_dir + 'log_females/' + hp.model_name)
            best = loss_val
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
            if num_bad_epochs == hp.patience:
                break

    print('Done')

if __name__ == '__main__':
    main()
