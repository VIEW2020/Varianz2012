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

    ax.legend(['CVD Equations', 'Deep Learning'])

    lim = max([df_cox['EVENT_PERC'].max(), df_cox['RISK'].max(), df_cml['EVENT_PERC'].max(), df_cml['RISK'].max()])+0.5
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
    
    ax.scatter(df_cox['QUANTILE'].astype(float)-0.1, df_cox['EVENT_PERC_TOTAL'], c='lightcoral')
    ax.scatter(df_cml['QUANTILE'].astype(float)+0.1, df_cml['EVENT_PERC_TOTAL'], c='cornflowerblue')

    ax.legend(['CVD Equations', 'Deep Learning'])
    
    ax.set_xlabel('Deciles of predicted risk')
    ax.set_ylabel('Proportion of all CVD events')
    
    return(ax)


def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    means = np.load(hp.data_pp_dir + 'means_' + hp.gender + '.npz')
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
    df_cox['RISK'] = feather.read_dataframe(hp.results_dir + 'df_cox_' + hp.gender + '.feather')['RISK']
    df_cml['RISK'] = 0
    for fold in range(hp.num_folds):
        df_cml.loc[data['fold'] == fold, 'RISK'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '.feather')['ENSEMBLE'].values
    
    # remove validation data
    df_cox = df_cox[data['fold'] != 99]
    df_cml = df_cml[data['fold'] != 99]

    ################################################################################################

    print('Plot all...')
    plt.figure()
    calibration_plot(df_cox, df_cml)
    plt.title('Calibration: Men') if hp.gender == 'males' else plt.title('Calibration: Women')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_all.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox, df_cml)
    plt.title('Discrimination: Men') if hp.gender == 'males' else plt.title('Discrimination: Women')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_all.pdf')
    plt.close()
    
    ################################################################################################
    
    print('Plot by age...')
    #30-44
    condition = (df_cox['nhi_age'] >= 30) & (df_cox['nhi_age'] < 45)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men 30-44 years') if hp.gender == 'males' else plt.title('Calibration: Women 30-44 years')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_age_30_44.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men 30-44 years') if hp.gender == 'males' else plt.title('Discrimination: Women 30-44 years')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_age_30_44.pdf')    
    plt.close()
    
    #45-59
    condition = (df_cox['nhi_age'] >= 45) & (df_cox['nhi_age'] < 60)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men 45-59 years') if hp.gender == 'males' else plt.title('Calibration: Women 45-59 years')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_age_45_59.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men 45-59 years') if hp.gender == 'males' else plt.title('Discrimination: Women 45-59 years')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_age_45_59.pdf')
    plt.close()
    
    #60-74
    condition = (df_cox['nhi_age'] >= 60) & (df_cox['nhi_age'] < 75)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men 60-74 years') if hp.gender == 'males' else plt.title('Calibration: Women 60-74 years')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_age_60_74.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men 60-74 years') if hp.gender == 'males' else plt.title('Discrimination: Women 60-74 years')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_age_60_74.pdf')  
    plt.close()
    
    ################################################################################################
    
    print('Plot by ethnicity...')
    #Maori
    condition = df_cox['en_prtsd_eth_2'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Maori Men') if hp.gender == 'males' else plt.title('Calibration: Maori Women')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_eth_maori.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Maori Men') if hp.gender == 'males' else plt.title('Discrimination: Maori Women')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_eth_maori.pdf')    
    plt.close()

    #Pacific
    condition = df_cox['en_prtsd_eth_3'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Pacific Men') if hp.gender == 'males' else plt.title('Calibration: Pacific Women')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_eth_pacific.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Pacific Men') if hp.gender == 'males' else plt.title('Discrimination: Pacific Women')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_eth_pacific.pdf')      
    plt.close()

    #Indian
    condition = df_cox['en_prtsd_eth_43'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Indian Men') if hp.gender == 'males' else plt.title('Calibration: Indian Women')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_eth_indian.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Indian Men') if hp.gender == 'males' else plt.title('Discrimination: Indian Women')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_eth_indian.pdf')
    plt.close()

    #Other
    condition = df_cox['en_prtsd_eth_9'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men of Other Ethnicity') if hp.gender == 'males' else plt.title('Calibration: Women of Other Ethnicity')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_eth_other.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men of Other Ethnicity') if hp.gender == 'males' else plt.title('Discrimination: Women of Other Ethnicity')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_eth_other.pdf')
    plt.close()

    #NZ European
    condition = (~df_cox['en_prtsd_eth_2'].astype(bool)) & (~df_cox['en_prtsd_eth_3'].astype(bool)) & (~df_cox['en_prtsd_eth_43'].astype(bool)) & (~df_cox['en_prtsd_eth_9'].astype(bool)) 
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: NZ European Men') if hp.gender == 'males' else plt.title('Calibration: NZ European Women')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_eth_european.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: NZ European Men') if hp.gender == 'males' else plt.title('Discrimination: NZ European Women')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_eth_european.pdf')
    plt.close()

    ################################################################################################

    print('Plot by deprivation...')
    #1
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 1)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men Deprivation Q1') if hp.gender == 'males' else plt.title('Calibration: Women Deprivation Q1')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_deprivation_q1.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men Deprivation Q1') if hp.gender == 'males' else plt.title('Discrimination: Women Deprivation Q1')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_deprivation_q1.pdf')   
    plt.close()

    #2
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 2)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men Deprivation Q2') if hp.gender == 'males' else plt.title('Calibration: Women Deprivation Q2')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_deprivation_q2.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men Deprivation Q2') if hp.gender == 'males' else plt.title('Discrimination: Women Deprivation Q2')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_deprivation_q2.pdf')   
    plt.close()
    
    #3
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 3)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men Deprivation Q3') if hp.gender == 'males' else plt.title('Calibration: Women Deprivation Q3')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_deprivation_q3.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men Deprivation Q3') if hp.gender == 'males' else plt.title('Discrimination: Women Deprivation Q3')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_deprivation_q3.pdf')   
    plt.close()
    
    #4
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 4)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men Deprivation Q4') if hp.gender == 'males' else plt.title('Calibration: Women Deprivation Q4')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_deprivation_q4.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men Deprivation Q4') if hp.gender == 'males' else plt.title('Discrimination: Women Deprivation Q4')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_deprivation_q4.pdf')   
    plt.close()
    
    #5
    condition = (df_cox['en_nzdep_q'].round().astype(int) == 5)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men Deprivation Q5') if hp.gender == 'males' else plt.title('Calibration: Women Deprivation Q5')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_deprivation_q5.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men Deprivation Q5') if hp.gender == 'males' else plt.title('Discrimination: Women Deprivation Q5')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_deprivation_q5.pdf')   
    plt.close()

    ################################################################################################

    print('Plot by baseline medication...')
    #BPL
    condition = df_cox['ph_bp_lowering_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men with Baseline BPL Medications') if hp.gender == 'males' else plt.title('Calibration: Women with Baseline BPL Medications')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_med_bpl.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men with Baseline BPL Medications') if hp.gender == 'males' else plt.title('Discrimination: Women with Baseline BPL Medications')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_med_bpl.pdf')    
    plt.close()

    #No BPL
    condition = ~df_cox['ph_bp_lowering_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men without Baseline BPL Medications') if hp.gender == 'males' else plt.title('Calibration: Women without Baseline BPL Medications')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_med_no_bpl.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men without Baseline BPL Medications') if hp.gender == 'males' else plt.title('Discrimination: Women without Baseline BPL Medications')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_med_no_bpl.pdf')
    plt.close()

    #LL
    condition = df_cox['ph_lipid_lowering_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men with Baseline LL Medications') if hp.gender == 'males' else plt.title('Calibration: Women with Baseline LL Medications')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_med_ll.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men with Baseline LL Medications') if hp.gender == 'males' else plt.title('Discrimination: Women with Baseline LL Medications')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_med_ll.pdf')    
    plt.close()

    #No LL
    condition = ~df_cox['ph_lipid_lowering_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men without Baseline LL Medications') if hp.gender == 'males' else plt.title('Calibration: Women without Baseline LL Medications')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_med_no_ll.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men without Baseline LL Medications') if hp.gender == 'males' else plt.title('Discrimination: Women without Baseline LL Medications')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_med_no_ll.pdf')
    plt.close()

    #APL/AC
    condition = df_cox['ph_antiplat_anticoag_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men with Baseline APL/AC Medications') if hp.gender == 'males' else plt.title('Calibration: Women with Baseline APL/AC Medications')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_med_apl_ac.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men with Baseline APL/AC Medications') if hp.gender == 'males' else plt.title('Discrimination: Women with Baseline APL/AC Medications')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_med_apl_ac.pdf')    
    plt.close()

    #No APL/AC
    condition = ~df_cox['ph_antiplat_anticoag_prior_6mths'].astype(bool)
    print('Num people: ', sum(condition))
    df_cox_red = df_cox.loc[condition].copy()
    df_cml_red = df_cml.loc[condition].copy()
    
    plt.figure()
    calibration_plot(df_cox_red, df_cml_red)
    plt.title('Calibration: Men without Baseline APL/AC Medications') if hp.gender == 'males' else plt.title('Calibration: Women without Baseline APL/AC Medications')
    plt.savefig(hp.plots_dir + 'calibration_' + hp.gender + '_med_no_apl_ac.pdf')
    plt.close()

    plt.figure()
    discrimination_plot(df_cox_red, df_cml_red)
    plt.title('Discrimination: Men without Baseline APL/AC Medications') if hp.gender == 'males' else plt.title('Discrimination: Women without Baseline APL/AC Medications')
    plt.savefig(hp.plots_dir + 'discrimination_' + hp.gender + '_med_no_apl_ac.pdf')
    plt.close()
    
    
if __name__ == '__main__':
    main()

