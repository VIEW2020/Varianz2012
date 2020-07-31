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

from hyperparameters import Hyperparameters
from utils import *

from pdb import set_trace as bp


def calibration_plot(df_cox, df_cml, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()

    num_people = len(df_cox.index)
    assert num_people == len(df_cml.index)
    
    df_cox['QUANTILE'] = pd.qcut(df_cox['RISK'], q=10, labels=list(range(10)))
    df_cml['QUANTILE'] = pd.qcut(df_cml['RISK'], q=10, labels=list(range(10)))
    
    df_cox = df_cox.groupby('QUANTILE').agg({'RISK':'mean', 'EVENT':'sum'})
    df_cml = df_cml.groupby('QUANTILE').agg({'RISK':'mean', 'EVENT':'sum'})

    df_cox['EVENT_PERC'] = df_cox['EVENT']/num_people*1000
    df_cml['EVENT_PERC'] = df_cml['EVENT']/num_people*1000
    
    df_cox.reset_index(inplace=True)
    df_cml.reset_index(inplace=True)
    
    ax.scatter(df_cox['EVENT_PERC'], df_cox['RISK'], c='lightcoral')
    ax.scatter(df_cml['EVENT_PERC'], df_cml['RISK'], c='cornflowerblue')

    ax.legend(['Cox PH', 'Deep Learning'])

    lim = max(ax.get_xlim(), ax.get_ylim())[1]
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.plot([0, lim], [0, lim], color = 'black', linewidth = 0.5)
    
    ax.set_xlabel('Observed events [%]')
    ax.set_ylabel('Mean predicted 5 year risk [%]')
    
    return(ax)


def discrimination_plot(df_cox, df_cml, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()

    num_people = len(df_cox.index)
    assert num_people == len(df_cml.index)
    
    df_cox['QUANTILE'] = pd.qcut(df_cox['RISK'], q=10, labels=list(range(10)))
    df_cml['QUANTILE'] = pd.qcut(df_cml['RISK'], q=10, labels=list(range(10)))
    
    df_cox = df_cox.groupby('QUANTILE').agg({'EVENT':'sum'})
    df_cml = df_cml.groupby('QUANTILE').agg({'EVENT':'sum'})

    df_cox['EVENT_PERC_TOTAL'] = df_cox['EVENT']/df_cox['EVENT'].sum()*100
    df_cml['EVENT_PERC_TOTAL'] = df_cml['EVENT']/df_cml['EVENT'].sum()*100
    
    df_cox.reset_index(inplace=True)
    df_cml.reset_index(inplace=True)
    
    ax.scatter(df_cox['QUANTILE'], df_cox['EVENT_PERC_TOTAL'], c='lightcoral')
    ax.scatter(df_cml['QUANTILE'], df_cml['EVENT_PERC_TOTAL'], c='cornflowerblue')

    ax.legend(['Cox PH', 'Deep Learning'])
    
    ax.set_xlabel('Deciles of predicted risk')
    ax.set_ylabel('Proportion of all CVD events')
    
    return(ax)


def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
        
    x = data['x_tst']
    time = data['time_tst']
    event = data['event_tst']
    
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    df_cox = pd.DataFrame(x, columns=cols_list)
    df_cox['TIME'] = time
    df_cox['EVENT'] = event
    
    df_cml = pd.DataFrame(x, columns=cols_list)
    df_cml['TIME'] = time
    df_cml['EVENT'] = event
    
    data = np.load(hp.results_dir + 'risk_cox_standard_' + hp.gender + '.npz')
    df_cox['RISK'] = data['risk']
    
    data = np.load(hp.results_dir + 'risk_matrix_' + hp.gender + '.npz')
    risk_matrix = data['risk_matrix']
    df_cml['RISK'] = risk_matrix.mean(axis=1)

    ################################################################################################

    print('Plot...')
    plt.figure()
    calibration_plot(df_cox, df_cml)
    plt.title('Calibration: Men') if hp.gender == 'males' else plt.title('Calibration: Women')
    plt.savefig(hp.results_dir + 'calibration_' + hp.gender + '.pdf')

    plt.figure()
    discrimination_plot(df_cox, df_cml)
    plt.title('Discrimination: Men') if hp.gender == 'males' else plt.title('Discrimination: Women')
    plt.savefig(hp.results_dir + 'discrimination_' + hp.gender + '.pdf')
    
    
    
if __name__ == '__main__':
    main()

