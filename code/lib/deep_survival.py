'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import numpy as np
import math
import pandas as pd
import pickle as pkl
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pdb import set_trace as bp


def sort_and_case_indices(x, time, event):
    """
    Sort data to allow for efficient sampling of people at risk.
    Time is in descending order, in case of ties non-events come first.
    In general, after sorting, if the index of A is smaller than the index of B,
    A is at risk when B experiences the event.
    To avoid sampling from ties, the column 'MAX_IDX_CONTROL' indicates the maximum
    index from which a case can be sampled.
    
    Args:
        x: input data
        time: time to event/censoring
        event: binary vector, 1 if the person experienced an event or 0 if censored
        
    Returns:
        sort_index: index to sort indices according to risk
        case_index: index to extract cases (on data sorted by sort_index!)
        max_idx_control: maximum index to sample a control for each case
    """
    # Sort
    df = pd.DataFrame({'TIME': time, 'EVENT': event.astype(bool)})
    df.sort_values(by=['TIME', 'EVENT'], ascending=[False, True], inplace=True)
    sort_index = df.index
    df.reset_index(drop=True, inplace=True)

    # Max idx for sampling controls (either earlier times or same time but no event)
    df['MAX_IDX_CONTROL'] = -1
    max_idx_control = -1
    prev_time = df.at[0, 'TIME']
    print('Computing MAX_IDX_CONTROL, time for a(nother) coffee...')
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not row['EVENT']:
            max_idx_control = i
        elif (prev_time > row['TIME']):
            max_idx_control = i-1
        df.at[i, 'MAX_IDX_CONTROL'] = max_idx_control
        prev_time = row['TIME']
    print('done')
    df_case = df[df['EVENT'] & (df['MAX_IDX_CONTROL']>=0)]
    case_index, max_idx_control = df_case.index, df_case['MAX_IDX_CONTROL'].values
    
    return sort_index, case_index, max_idx_control


class CoxPHLoss(torch.nn.Module):
    """
    Approximated Cox Proportional Hazards loss (negative partial likelihood 
        with 1 control sample)
    
    Args:
        risk_case: The predicted risk for people who experienced the event (batch_size,) 
        risk_control: The predicted risk for people who where at risk at the time 
            when the corresponding case experienced the event (batch_size,) 

    Returns:
        loss: Scalar loss
    """    
    def __init__(self):
        super(CoxPHLoss, self).__init__()
        
    def forward(self, risk_case, risk_control):
        loss = (F.softplus(risk_control - risk_case)).mean()
        return loss

    
def get_case_control(x_case, time_case, max_idx_control, code_case, month_case, diagt_case, x, code, month, diagt, hp):
    control_idx = (np.random.uniform(size=(x_case.shape[0],)) * max_idx_control.numpy()).astype(int)
    x_control = x[control_idx]
    x_cc = torch.cat([x_case, x_control]).to(hp.device)
    if hp.nonprop_hazards:
        time_cc = torch.unsqueeze(torch.cat([time_case, time_case]), 1).float().to(hp.device) # time is the same for cases and controls
    else:
        time_cc = None
    code_control = code[control_idx]
    code_cc = torch.cat([code_case, code_control]).to(hp.device)
    month_control = month[control_idx]
    month_cc = torch.cat([month_case, month_control]).to(hp.device)
    diagt_control = diagt[control_idx]
    diagt_cc = torch.cat([diagt_case, diagt_control]).to(hp.device)    
    return x_cc, time_cc, code_cc, month_cc, diagt_cc


def trn(trn_loader, x_trn, code_trn, month_trn, diagt_trn, model, criterion, optimizer, hp):
    model.train()
    for batch_idx, (x_case, time_case, max_idx_control, code_case, month_case, diagt_case) in enumerate(tqdm(trn_loader)):
        # Get controls
        x_cc, time_cc, code_cc, month_cc, diagt_cc = get_case_control(x_case, time_case, max_idx_control, code_case, month_case, diagt_case, x_trn, code_trn, month_trn, diagt_trn, hp)
        
        # Optimise
        optimizer.zero_grad()
        risk_case, risk_control = model(x_cc, code_cc, month_cc, diagt_cc, time_cc).chunk(2)
        loss = criterion(risk_case, risk_control)
        loss.backward()
        optimizer.step()

        
def val(val_loader, x_val, code_val, month_val, diagt_val, model, criterion, epoch, hp):
    loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x_case, time_case, max_idx_control, code_case, month_case, diagt_case) in enumerate(tqdm(val_loader)):
            # Get controls
            x_cc, time_cc, code_cc, month_cc, diagt_cc = get_case_control(x_case, time_case, max_idx_control, code_case, month_case, diagt_case, x_val, code_val, month_val, diagt_val, hp)
            
            # Compute Loss
            risk_case, risk_control = model(x_cc, code_cc, month_cc, diagt_cc, time_cc).chunk(2)
            loss += criterion(risk_case, risk_control)

        loss_norm = loss.item()/len(val_loader)
        print('Epoch: {} Loss: {:.6f}'.format(epoch, loss_norm))
        return loss_norm


