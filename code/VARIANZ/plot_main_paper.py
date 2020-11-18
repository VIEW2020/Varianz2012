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

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('font',**{'family':'Times New Roman', 'size':18})
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.linewidth'] = 1

from hyperparameters import Hyperparameters
from utils import *
from EvalSurv import EvalSurv
from plot import calibration_plot, discrimination_plot

from pdb import set_trace as bp


def main():
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,14))
    fig_cal, ax_cal = plt.subplots(nrows=3, ncols=2, figsize=(16,21))
    
    for i, gender in enumerate(['females', 'males']):
    
        # Load data
        print('Load data...')
        hp = Hyperparameters()
        data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
        means = np.load(hp.data_pp_dir + 'means_' + hp.gender + '.npz')
        x = data['x']
        time = data['time']
        event = data['event']
        cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
        
        # restore original age and en_nzdep_q before centering
        x[:, cols_list.index('nhi_age')] += means['mean_age']
        x[:, cols_list.index('en_nzdep_q')] += means['mean_nzdep']
        
        df_cox = pd.DataFrame(x, columns=cols_list)
        df_cox['TIME'] = time
        df_cox['EVENT'] = event
        
        df_cml = pd.DataFrame(x, columns=cols_list)
        df_cml['TIME'] = time
        df_cml['EVENT'] = event
        
        # load predicted risk
        lph_matrix_cox = np.zeros((df_cox.shape[0], hp.num_folds))
        lph_matrix_cml = np.zeros((df_cml.shape[0], hp.num_folds))
        for fold in range(hp.num_folds):
            for swap in range(2):
                print('Fold: {} Swap: {}'.format(fold, swap))
                idx = (data['fold'][:, fold] == swap)
                lph_matrix_cox[idx, fold] = feather.read_dataframe(hp.results_dir + 'df_cox_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '.feather')['LPH']
                lph_matrix_cml[idx, fold] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '.feather')['LPH']
        df_cox['LPH'] = lph_matrix_cox.mean(axis=1)
        df_cml['LPH'] = lph_matrix_cml.mean(axis=1)
        
        # remove validation data
        idx = (data['fold'][:, fold] != 99)
        df_cox = df_cox[idx].reset_index(drop=True)
        df_cml = df_cml[idx].reset_index(drop=True)
        es_cox = EvalSurv(df_cox.copy())
        es_cml = EvalSurv(df_cml.copy())
        
        df_cox['RISK_PERC'] = es_cox.get_risk_perc(1826)
        df_cml['RISK_PERC'] = es_cml.get_risk_perc(1826)

        ################################################################################################

        ax_plt = ax[i][0]
        calibration_plot(df_cox, df_cml, ax_plt)
        ax_plt.title.set_text('Calibration: Men') if gender == 'males' else ax_plt.title.set_text('Calibration: Women')

        ax_plt = ax[i][1]
        discrimination_plot(df_cox, df_cml, ax_plt)
        ax_plt.title.set_text('Discrimination: Men') if gender == 'males' else ax_plt.title.set_text('Discrimination: Women')
                
        ################################################################################################
        
        #30-44
        condition = (df_cox['nhi_age'] >= 30) & (df_cox['nhi_age'] < 45)
        print('Num people: ', sum(condition))
        df_cox_red = df_cox.loc[condition].copy()
        df_cml_red = df_cml.loc[condition].copy()
        
        ax_plt = ax_cal[0][i]
        calibration_plot(df_cox_red, df_cml_red, ax_plt)
        ax_plt.title.set_text('Calibration: Men 30-44 years') if gender == 'males' else ax_plt.title.set_text('Calibration: Women 30-44 years')

        #Maori
        condition = df_cox['en_prtsd_eth_2'].astype(bool)
        print('Num people: ', sum(condition))
        df_cox_red = df_cox.loc[condition].copy()
        df_cml_red = df_cml.loc[condition].copy()
        
        ax_plt = ax_cal[1][i]
        calibration_plot(df_cox_red, df_cml_red, ax_plt)
        ax_plt.title.set_text('Calibration: Maori Men') if gender == 'males' else ax_plt.title.set_text('Calibration: Maori Women')
        
        #5
        condition = (df_cox['en_nzdep_q'].round().astype(int) == 5)
        print('Num people: ', sum(condition))
        df_cox_red = df_cox.loc[condition].copy()
        df_cml_red = df_cml.loc[condition].copy()
        
        ax_plt = ax_cal[2][i]
        calibration_plot(df_cox_red, df_cml_red, ax_plt)
        ax_plt.title.set_text('Calibration: Men Deprivation Q5') if gender == 'males' else ax_plt.title.set_text('Calibration: Women Deprivation Q5')
        
        ################################################################################################
 
    fig.tight_layout()
    fig.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig.savefig(hp.plots_dir + 'all.png')    
    
    fig_cal.tight_layout()
    fig_cal.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig_cal.savefig(hp.plots_dir + 'examples_calibration.png')
        
    
if __name__ == '__main__':
    main()

