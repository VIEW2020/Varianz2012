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
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import tqdm

from hyperparameters import Hyperparameters
from utils import *
from EvalSurv import EvalSurv

from pdb import set_trace as bp
    

def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    time = data['time']
    event = data['event']
    
    df = pd.DataFrame({'TIME': data['time'], 'EVENT': data['event']})
    df_cox = df.copy()
    df_cml = df.copy()

    # load log partial hazards
    df_cox['LPH'] = feather.read_dataframe(hp.results_dir + 'df_cox_' + hp.gender + '.feather')['LPH']
    df_cml['LPH'] = 0
    for fold in range(hp.num_folds):
        df_cml.loc[data['fold'] == fold, 'LPH'] = feather.read_dataframe(hp.results_dir + 'df_lph_' + hp.gender + '_fold_' + str(fold) + '.feather')['ENSEMBLE'].values
    
    # remove validation data
    df_cox = df_cox[data['fold'] != 99]
    df_cml = df_cml[data['fold'] != 99]
    
    # evaluate
    es_cox = EvalSurv(df_cox)
    es_cml = EvalSurv(df_cml)

    ################################################################################################

    d_index_cox, lCI_cox, uCI_cox = es_cox.D_index()
    print('D-index Cox (95% CI): {:.5}'.format(d_index_cox), ' ({:.5}'.format(lCI_cox), ',  {:.5}'.format(uCI_cox), ")")
    print('R-squared(D) Cox: {:.5}'.format(es_cox.R_squared_D()))
    print('Concordance Cox: {:.5}'.format(es_cox.concordance_index()))
    print('IBS Cox: {:.5}'.format(es_cox.integrated_brier_score()))
    print('AUC Cox: {:.5}'.format(es_cox.auc(1826)))

    d_index_cml, lCI_cml, uCI_cml = es_cml.D_index()
    print('D-index ML (95% CI): {:.5}'.format(d_index_cml), ' ({:.5}'.format(lCI_cml), ',  {:.5}'.format(uCI_cml), ")")
    print('R-squared(D) ML: {:.5}'.format(es_cml.R_squared_D()))
    print('Concordance ML: {:.5}'.format(es_cml.concordance_index()))
    print('IBS ML: {:.5}'.format(es_cml.integrated_brier_score()))
    print('AUC ML: {:.5}'.format(es_cml.auc(1826)))
    
if __name__ == '__main__':
    main()

