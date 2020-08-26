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
mpl.rc('font',**{'family':'sans-serif', 'sans-serif':['DejaVu Sans'], 'size':18})
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.linewidth'] = 1

from hyperparameters import Hyperparameters
from utils import *
from plot import calibration_plot, discrimination_plot

from pdb import set_trace as bp


def main():
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,12))
    fig_cal, ax_cal = plt.subplots(nrows=3, ncols=2, figsize=(16,18))
    
    for i, gender in enumerate([females, males]):
    
        # Load data
        print('Load data...')
        hp = Hyperparameters()
        data = np.load(hp.data_pp_dir + 'data_arrays_' + gender + '.npz')
        means = np.load(hp.data_pp_dir + 'means_' + gender + '.npz')
        x = data['x']
        event = data['event']
        cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
        
        # restore original age and en_nzdep_q before centering
        x[:, cols_list.index('nhi_age')] += means['mean_age']
        x[:, cols_list.index('en_nzdep_q')] += means['mean_nzdep']
        
        df_cox = pd.DataFrame(x, columns=cols_list)
        df_cox['EVENT'] = event
        
        df_cml = pd.DataFrame(x, columns=cols_list)
        df_cml['EVENT'] = event
        
        # load predicted risk
        df_cox['RISK_PERC'] = feather.read_dataframe(hp.results_dir + 'df_cox_' + gender + '.feather')['RISK_PERC']
        df_cml['RISK_PERC'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + gender + '.feather')['RISK_PERC']
        
        # remove validation data
        df_cox = df_cox[data['fold'] != 99]
        df_cml = df_cml[data['fold'] != 99]

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
        
        ax_plt = ax_cal[0][i]
        calibration_plot(df_cox_red, df_cml_red, ax_plt)
        ax_plt.title.set_text('Calibration: Maori Men') if gender == 'males' else ax_plt.title.set_text('Calibration: Maori Women')
        
        #5
        condition = (df_cox['en_nzdep_q'].round().astype(int) == 5)
        print('Num people: ', sum(condition))
        df_cox_red = df_cox.loc[condition].copy()
        df_cml_red = df_cml.loc[condition].copy()
        
        ax_plt = ax_cal[2][0]
        calibration_plot(df_cox_red, df_cml_red, ax_plt)
        ax_plt.title.set_text('Calibration: Men Deprivation Q5') if gender == 'males' else ax_plt.title.set_text('Calibration: Women Deprivation Q5')
        
        ################################################################################################
 
    fig.tight_layout()
    fig.subplots_adjust(wspace = 0.3)
    fig.savefig(hp.plots_dir + 'all.png')    
    
    fig_cal.tight_layout()
    fig_cal.subplots_adjust(wspace = 0.3)
    fig_cal.savefig(hp.plots_dir + 'examples_calibration.png')
        
    
if __name__ == '__main__':
    main()

