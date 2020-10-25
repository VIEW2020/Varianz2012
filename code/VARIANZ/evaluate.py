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
import statsmodels.stats.api as sms

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

    # evaluation vectors
    d_index_vec_cox = np.zeros((hp.num_folds, 2))
    r2_vec_cox = np.zeros((hp.num_folds, 2))
    concordance_vec_cox = np.zeros((hp.num_folds, 2))
    ibs_vec_cox = np.zeros((hp.num_folds, 2))
    auc_vec_cox = np.zeros((hp.num_folds, 2))

    d_index_vec_cml = np.zeros((hp.num_folds, 2))
    r2_vec_cml = np.zeros((hp.num_folds, 2))
    concordance_vec_cml = np.zeros((hp.num_folds, 2))
    ibs_vec_cml = np.zeros((hp.num_folds, 2))
    auc_vec_cml = np.zeros((hp.num_folds, 2))
    
    print('Evaluate on each fold...')
    for fold in range(hp.num_folds):
        for swap in range(2):
            print('Fold: {} Swap: {}'.format(fold, swap))
            
            idx = (data['fold'][:, fold] == swap)
            df_fold = df[idx].reset_index(drop=True)
    
            df_cox = df_fold.copy()
            df_cml = df_fold.copy()

            # load log partial hazards
            df_cox['LPH'] = feather.read_dataframe(hp.results_dir + 'df_cox_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '.feather')['LPH']
            df_cml['LPH'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '.feather')['LPH']
    
            ################################################################################################
            
            # baseline survival
            # es_cox = EvalSurv(df_cox.copy())
            # es_cml = EvalSurv(df_cml.copy())    
            # print('Base survival Cox: {:.5}'.format(es_cox.get_base_surv(1826)))
            # print('Base survival CML: {:.5}'.format(es_cml.get_base_surv(1826)))
            # return
                            
            es_cox = EvalSurv(df_cox.copy())
            es_cml = EvalSurv(df_cml.copy())

            r2_vec_cox[fold, swap] = es_cox.R_squared_D()
            d_index_vec_cox[fold, swap], _ = es_cox.D_index()
            concordance_vec_cox[fold, swap] = es_cox.concordance_index()
            ibs_vec_cox[fold, swap] = es_cox.integrated_brier_score()
            auc_vec_cox[fold, swap] = es_cox.auc(1826)
            
            r2_vec_cml[fold, swap] = es_cml.R_squared_D()
            d_index_vec_cml[fold, swap], _ = es_cml.D_index()
            concordance_vec_cml[fold, swap] = es_cml.concordance_index()
            ibs_vec_cml[fold, swap] = es_cml.integrated_brier_score()
            auc_vec_cml[fold, swap] = es_cml.auc(1826)

    print('Save...')
    np.savez(hp.results_dir + 'eval_vecs_' + hp.gender + '.npz', 
             r2_vec_cox=r2_vec_cox, d_index_vec_cox=d_index_vec_cox, concordance_vec_cox=concordance_vec_cox, ibs_vec_cox=ibs_vec_cox, auc_vec_cox=auc_vec_cox, 
             r2_vec_cml=r2_vec_cml, d_index_vec_cml=d_index_vec_cml, concordance_vec_cml=concordance_vec_cml, ibs_vec_cml=ibs_vec_cml, auc_vec_cml=auc_vec_cml)

    r2_vec_cox=np.reshape(r2_vec_cox,-1)
    d_index_vec_cox=np.reshape(d_index_vec_cox,-1)
    concordance_vec_cox=np.reshape(concordance_vec_cox,-1)
    ibs_vec_cox=np.reshape(ibs_vec_cox,-1)
    auc_vec_cox=np.reshape(auc_vec_cox,-1)
    r2_vec_cml=np.reshape(r2_vec_cml,-1)
    d_index_vec_cml=np.reshape(d_index_vec_cml,-1)
    concordance_vec_cml=np.reshape(concordance_vec_cml,-1)
    ibs_vec_cml=np.reshape(ibs_vec_cml,-1)
    auc_vec_cml=np.reshape(auc_vec_cml,-1)

    r2_mean, (r2_lci, r2_uci) = r2_vec_cox.mean(), sms.DescrStatsW(r2_vec_cox).tconfint_mean()
    print('R-squared(D) Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(r2_mean, r2_lci, r2_uci))
    d_index_mean, (d_index_lci, d_index_uci) = d_index_vec_cox.mean(), sms.DescrStatsW(d_index_vec_cox).tconfint_mean()
    print('D-index Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(d_index_mean, d_index_lci, d_index_uci))
    concordance_mean, (concordance_lci, concordance_uci) = concordance_vec_cox.mean(), sms.DescrStatsW(concordance_vec_cox).tconfint_mean()
    print('Concordance Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(concordance_mean, concordance_lci, concordance_uci))
    ibs_mean, (ibs_lci, ibs_uci) = ibs_vec_cox.mean(), sms.DescrStatsW(ibs_vec_cox).tconfint_mean()
    print('IBS Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(ibs_mean, ibs_lci, ibs_uci))
    auc_mean, (auc_lci, auc_uci) = auc_vec_cox.mean(), sms.DescrStatsW(auc_vec_cox).tconfint_mean()
    print('AUC Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(auc_mean, auc_lci, auc_uci))

    r2_mean, (r2_lci, r2_uci) = r2_vec_cml.mean(), sms.DescrStatsW(r2_vec_cml).tconfint_mean()
    print('R-squared(D) CML (95% CI): {:.3} ({:.3}, {:.3})'.format(r2_mean, r2_lci, r2_uci))
    d_index_mean, (d_index_lci, d_index_uci) = d_index_vec_cml.mean(), sms.DescrStatsW(d_index_vec_cml).tconfint_mean()
    print('D-index CML (95% CI): {:.3} ({:.3}, {:.3})'.format(d_index_mean, d_index_lci, d_index_uci))
    concordance_mean, (concordance_lci, concordance_uci) = concordance_vec_cml.mean(), sms.DescrStatsW(concordance_vec_cml).tconfint_mean()
    print('Concordance CML (95% CI): {:.3} ({:.3}, {:.3})'.format(concordance_mean, concordance_lci, concordance_uci))
    ibs_mean, (ibs_lci, ibs_uci) = ibs_vec_cml.mean(), sms.DescrStatsW(ibs_vec_cml).tconfint_mean()
    print('IBS CML (95% CI): {:.3} ({:.3}, {:.3})'.format(ibs_mean, ibs_lci, ibs_uci))
    auc_mean, (auc_lci, auc_uci) = auc_vec_cml.mean(), sms.DescrStatsW(auc_vec_cml).tconfint_mean()
    print('AUC CML (95% CI): {:.3} ({:.3}, {:.3})'.format(auc_mean, auc_lci, auc_uci))

    
if __name__ == '__main__':
    main()

