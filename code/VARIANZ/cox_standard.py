'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import sys
#sys.path.append('..\lib\\')
sys.path.append('../lib/')

import numpy as np
import pandas as pd
from deep_survival import *
import feather
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from pycox.evaluation import EvalSurv
from lifelines import CoxPHFitter
from hyperparameters import Hyperparameters as hp

from pdb import set_trace as bp


def main():
    # Load data
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays.npz')
    x_trn = data['x_trn']
    time_trn = data['time_trn']
    event_trn = data['event_trn']
    codes_trn = data['codes_trn']
    x_tst = data['x_tst']
    time_tst = data['time_tst']
    event_tst = data['event_tst']

    # rel_idx = [0,1,2,9,10,11,12,13,14,15] # exclude history covariates to avoid collinearity with individual codes
    # x_trn = x_trn[:, rel_idx]
    # x_tst = x_tst[:, rel_idx]

    # relevant_code = 2484
    for relevant_code in [910,923,2253,1738,1737,908,1133,2525,2842,2828,1484,2899,1860,1567,825,869,1631,870]:
        relevant_code_present = (codes_trn == relevant_code).max(axis=1)
        print('N codes: {}'.format(sum(relevant_code_present)))
        
        event_relevant_code    = event_trn[relevant_code_present]
        event_no_relevant_code = event_trn[~relevant_code_present]
        OR = (sum(event_relevant_code)/sum(~event_relevant_code))/(sum(event_no_relevant_code)/sum(~event_no_relevant_code))
        print('OR: {}'.format(OR))
    
    # df = pd.DataFrame(x_trn)
    # df['CODE'] = relevant_code_present.astype(int)
    # df['TIME'] = time_trn
    # df['EVENT'] = event_trn

    ###################################################################
    # print('Fitting...')
    # cph = CoxPHFitter()
    # cph.fit(df, duration_col='TIME', event_col='EVENT')
    # cph.print_summary()
    # print('done')

    #Prediction
    # surv = cph.predict_survival_function(x_tst)
    
    #Evaluation
    # ev = EvalSurv(surv, time_tst, event_tst, censor_surv='km')
    # concordance = ev.concordance_td()
    # print('Concordance: {:.6f}'.format(concordance))
    # time_grid = np.linspace(time_tst.min(), time_tst.max(), 100)
    # brier = ev.integrated_brier_score(time_grid)
    # print('Brier score: {:.6f}'.format(brier))
    # nbll = ev.integrated_nbll(time_grid)
    # print('NBLL: {:.6f}'.format(nbll))

    # Concordance: 0.770607
    # Brier score: 0.017257
    # NBLL: 0.077004



if __name__ == '__main__':
    main()
