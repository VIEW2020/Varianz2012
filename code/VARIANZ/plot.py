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

from pdb import set_trace as bp


def calibration_plot(df_cox, df_cml, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()

    num_people = len(df_cox.index)
    assert num_people == len(df_cml.index)
    
    df_cox['QUANTILE'] = pd.qcut(df_cox['RISK_PERC'], q=10, labels=list(range(10)))
    df_cml['QUANTILE'] = pd.qcut(df_cml['RISK_PERC'], q=10, labels=list(range(10)))
    
    df_cox = df_cox.groupby('QUANTILE').agg({'RISK_PERC':'mean', 'EVENT':'sum'})
    df_cml = df_cml.groupby('QUANTILE').agg({'RISK_PERC':'mean', 'EVENT':'sum'})

    df_cox['EVENT_PERC'] = df_cox['EVENT']/num_people*1000
    df_cml['EVENT_PERC'] = df_cml['EVENT']/num_people*1000
    
    df_cox.reset_index(inplace=True)
    df_cml.reset_index(inplace=True)
    
    ax.scatter(df_cox['EVENT_PERC'], df_cox['RISK_PERC'], c='lightcoral')
    ax.scatter(df_cml['EVENT_PERC'], df_cml['RISK_PERC'], c='cornflowerblue')

    ax.legend(['CVD Equations', 'Deep Learning'])

    lim = max([df_cox['EVENT_PERC'].max(), df_cox['RISK_PERC'].max(), df_cml['EVENT_PERC'].max(), df_cml['RISK_PERC'].max()])+0.5
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.plot([0, lim], [0, lim], color = 'black', linewidth = 1)
    
    ax.set_xlabel('Observed events [%]')
    ax.set_ylabel('Mean predicted 5 year risk [%]')
    
    return(ax)


def discrimination_plot(df_cox, df_cml, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()

    num_people = len(df_cox.index)
    assert num_people == len(df_cml.index)
    
    df_cox['QUANTILE'] = pd.qcut(df_cox['RISK_PERC'], q=10, labels=list(range(10)))
    df_cml['QUANTILE'] = pd.qcut(df_cml['RISK_PERC'], q=10, labels=list(range(10)))
    
    df_cox = df_cox.groupby('QUANTILE').agg({'EVENT':'sum'})
    df_cml = df_cml.groupby('QUANTILE').agg({'EVENT':'sum'})

    df_cox['EVENT_PERC_TOTAL'] = df_cox['EVENT']/df_cox['EVENT'].sum()*100
    df_cml['EVENT_PERC_TOTAL'] = df_cml['EVENT']/df_cml['EVENT'].sum()*100
    
    df_cox.reset_index(inplace=True)
    df_cml.reset_index(inplace=True)
    
    ax.scatter(df_cox['QUANTILE'].astype(float)-0.1, df_cox['EVENT_PERC_TOTAL'], c='lightcoral')
    ax.scatter(df_cml['QUANTILE'].astype(float)+0.1, df_cml['EVENT_PERC_TOTAL'], c='cornflowerblue')

    ax.legend(['CVD Equations', 'Deep Learning'])
    
    ax.set_xlabel('Deciles of predicted risk')
    ax.set_ylabel('Proportion of all CVD events [%]')
    
    return(ax)


def main():
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

    print('Plot all...')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,7))

    ax_plt = ax[0]
    calibration_plot(df_cox, df_cml, ax_plt)
    ax_plt.title.set_text('Calibration: Men') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women')

    ax_plt = ax[1]
    discrimination_plot(df_cox, df_cml, ax_plt)
    ax_plt.title.set_text('Discrimination: Men') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig.savefig(hp.plots_dir + hp.gender + '_all.png')
    plt.close()
    
    ################################################################################################
    
    print('Plot by age...')
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16,21))
    
    #30-44
    condition = (df_cox['nhi_age'] >= 30) & (df_cox['nhi_age'] < 45)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax[0][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men 30-44 years') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women 30-44 years')

    ax_plt = ax[0][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men 30-44 years') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women 30-44 years')
    
    #45-59
    condition = (df_cox['nhi_age'] >= 45) & (df_cox['nhi_age'] < 60)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax[1][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men 45-59 years') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women 45-59 years')

    ax_plt = ax[1][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men 45-59 years') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women 45-59 years')
    
    #60-74
    condition = (df_cox['nhi_age'] >= 60) & (df_cox['nhi_age'] < 75)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax[2][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men 60-74 years') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women 60-74 years')

    ax_plt = ax[2][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men 60-74 years') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women 60-74 years')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig.savefig(hp.plots_dir + hp.gender + '_age.png')
    plt.close()

    ################################################################################################
    
    print('Plot by ethnicity...')
    fig_cal, ax_cal = plt.subplots(nrows=3, ncols=2, figsize=(16,21))
    fig_dis, ax_dis = plt.subplots(nrows=3, ncols=2, figsize=(16,21))
    
    #Maori
    condition = df_cox['en_prtsd_eth_2'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[0][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Maori Men') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Maori Women')

    ax_plt = ax_dis[0][0]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Maori Men') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Maori Women')

    #Pacific
    condition = df_cox['en_prtsd_eth_3'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[0][1]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Pacific Men') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Pacific Women')

    ax_plt = ax_dis[0][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Pacific Men') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Pacific Women')

    #Indian
    condition = df_cox['en_prtsd_eth_43'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[1][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Indian Men') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Indian Women')

    ax_plt = ax_dis[1][0]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Indian Men') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Indian Women')

    #Other
    condition = df_cox['en_prtsd_eth_9'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[1][1]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men of Other Ethnicity') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women of Other Ethnicity')

    ax_plt = ax_dis[1][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men of Other Ethnicity') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women of Other Ethnicity')

    #NZ European
    condition = (~df_cox['en_prtsd_eth_2'].astype(bool)) & (~df_cox['en_prtsd_eth_3'].astype(bool)) & (~df_cox['en_prtsd_eth_43'].astype(bool)) & (~df_cox['en_prtsd_eth_9'].astype(bool)) 
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[2][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: European Men') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: European Women')

    ax_plt = ax_dis[2][0]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: European Men') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: European Women')

    ax_cal[2, 1].axis('off')
    ax_dis[2, 1].axis('off')
    fig_cal.tight_layout()
    fig_cal.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig_cal.savefig(hp.plots_dir + hp.gender + '_ethnicity_calibration.png')
    fig_dis.tight_layout()
    fig_dis.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig_dis.savefig(hp.plots_dir + hp.gender + '_ethnicity_discrimination.png')
    plt.close()

    ################################################################################################

    print('Plot by deprivation...')
    fig_cal, ax_cal = plt.subplots(nrows=3, ncols=2, figsize=(16,21))
    fig_dis, ax_dis = plt.subplots(nrows=3, ncols=2, figsize=(16,21))
    
    #1
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 1)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[0][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men Deprivation Q1') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women Deprivation Q1')

    ax_plt = ax_dis[0][0]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men Deprivation Q1') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women Deprivation Q1')

    #2
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 2)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[0][1]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men Deprivation Q2') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women Deprivation Q2')

    ax_plt = ax_dis[0][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men Deprivation Q2') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women Deprivation Q2')
    
    #3
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 3)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[1][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men Deprivation Q3') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women Deprivation Q3')

    ax_plt = ax_dis[1][0]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men Deprivation Q3') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women Deprivation Q3')
    
    #4
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 4)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[1][1]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men Deprivation Q4') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women Deprivation Q4')

    ax_plt = ax_dis[1][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men Deprivation Q4') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women Deprivation Q4')
    
    #5
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 5)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[2][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men Deprivation Q5') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women Deprivation Q5')

    ax_plt = ax_dis[2][0]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men Deprivation Q5') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women Deprivation Q5')

    ax_cal[2, 1].axis('off')
    ax_dis[2, 1].axis('off')
    fig_cal.tight_layout()
    fig_cal.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig_cal.savefig(hp.plots_dir + hp.gender + '_deprivation_calibration.png')
    fig_dis.tight_layout()
    fig_dis.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig_dis.savefig(hp.plots_dir + hp.gender + '_deprivation_discrimination.png')
    plt.close()

    ################################################################################################

    print('Plot by medication...')
    fig_cal, ax_cal = plt.subplots(nrows=3, ncols=2, figsize=(16,21))
    fig_dis, ax_dis = plt.subplots(nrows=3, ncols=2, figsize=(16,21))    
    
    #BPL
    condition = df_cox['ph_bp_lowering_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[0][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men with BPL Meds') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women with BPL Meds')

    ax_plt = ax_dis[0][0]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men with BPL Meds') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women with BPL Meds')

    #No BPL
    condition = ~df_cox['ph_bp_lowering_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[0][1]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men without BPL Meds') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women without BPL Meds')

    ax_plt = ax_dis[0][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men without BPL Meds') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women without BPL Meds')

    #LL
    condition = df_cox['ph_lipid_lowering_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[1][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men with LL Meds') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women with LL Meds')

    ax_plt = ax_dis[1][0]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men with LL Meds') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women with LL Meds')

    #No LL
    condition = ~df_cox['ph_lipid_lowering_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[1][1]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men without LL Meds') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women without LL Meds')

    ax_plt = ax_dis[1][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men without LL Meds') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women without LL Meds')

    #APL/AC
    condition = df_cox['ph_antiplat_anticoag_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[2][0]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men with APL/AC Meds') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women with APL/AC Meds')

    ax_plt = ax_dis[2][0]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men with APL/AC Meds') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women with APL/AC Meds')

    #No APL/AC
    condition = ~df_cox['ph_antiplat_anticoag_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    ax_plt = ax_cal[2][1]
    calibration_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Calibration: Men without APL/AC Meds') if hp.gender == 'males' else ax_plt.title.set_text('Calibration: Women without APL/AC Meds')

    ax_plt = ax_dis[2][1]
    discrimination_plot(df_cox_red, df_cml_red, ax_plt)
    ax_plt.title.set_text('Discrimination: Men without APL/AC Meds') if hp.gender == 'males' else ax_plt.title.set_text('Discrimination: Women without APL/AC Meds')
    
    fig_cal.tight_layout()
    fig_cal.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig_cal.savefig(hp.plots_dir + hp.gender + '_medication_calibration.png')
    fig_dis.tight_layout()
    fig_dis.subplots_adjust(wspace = 0.3, hspace = 0.3)
    fig_dis.savefig(hp.plots_dir + hp.gender + '_medication_discrimination.png')
    plt.close()

    
if __name__ == '__main__':
    main()

