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
from sklearn import preprocessing
import lifelines

from pdb import set_trace as bp


def main():
    # Load data
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    x_trn = data['x_trn']
    time_trn = data['time_trn']
    event_trn = data['event_trn']
    codes_trn = data['codes_trn']
    x_tst = data['x_tst']
    time_tst = data['time_tst']
    event_tst = data['event_tst']
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    df = pd.DataFrame(x_trn, columns=cols_list)
    df['TIME'] = time_trn
    df['EVENT'] = event_trn 

    #for  relevant_code in [1137]:
    ##for relevant_code in [499,652,1489,736,1043,1623,1008,1137,1042,498,696,1138,1403,1665,1677,770,623,1355,923,1579,1612,1172]:
        #relevant_code_present = (codes_trn == relevant_code).max(axis=1)
        #print('N codes: {}'.format(sum(relevant_code_present)))
        
        #event_relevant_code    = event_trn[relevant_code_present]
        #event_no_relevant_code = event_trn[~relevant_code_present]
        #OR = (sum(event_relevant_code)/sum(~event_relevant_code))/(sum(event_no_relevant_code)/sum(~event_no_relevant_code))
        #print('OR: {}'.format(OR))
    
    #df['CODE'] = relevant_code_present.astype(int)

    ###################################################################
    print('Fitting...')
    cph = CoxPHFitter()
    cph.fit(df, duration_col='TIME', event_col='EVENT', show_progress=True, step_size=0.5)
    cph.print_summary()
    print('done')
    
    ##Evaluation
    #surv = cph.predict_survival_function(x_tst)
    #ev = EvalSurv(surv, time_tst, event_tst, censor_surv='km')
    #concordance = ev.concordance_td()
    #print('Concordance: {:.6f}'.format(concordance))
    #time_grid = np.linspace(time_tst.min(), time_tst.max(), 100)
    #brier = ev.integrated_brier_score(time_grid)
    #print('Brier score: {:.6f}'.format(brier))
    #nbll = ev.integrated_nbll(time_grid)
    #print('NBLL: {:.6f}'.format(nbll))

    # Males
    # Concordance: 0.759711
    # Brier score: 0.021033
    # NBLL: 0.091169
    
    # Females
    # Concordance: 0.803357
    # Brier score: 0.013394
    # NBLL: 0.060757
    
    ###################################################################
    print('Predicting...')
    df_tst = pd.DataFrame(x_tst, columns=cols_list)
    df_tst['TIME'] = time_tst
    df_tst['EVENT'] = event_tst     
    base_surv = baseline_survival(df_tst, np.dot(x_tst-cph._norm_mean.values, cph.params_)).loc[1826]    
    df_tst['RISK'] = 100*(1-np.power(base_surv, np.exp(np.dot(x_tst-cph._norm_mean.values, cph.params_))))
    df_tst.to_feather(hp.plot_dir + 'df_tst_cox_' + hp.gender + '.feather')


if __name__ == '__main__':
    main()
