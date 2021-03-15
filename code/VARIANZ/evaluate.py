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

    #baseline survival CML
    df_cml = df.copy()
    lph_matrix = np.zeros((df_cml.shape[0], hp.num_folds))
    for fold in range(hp.num_folds):
        for swap in range(2):
            print('Fold: {} Swap: {}'.format(fold, swap))
            idx = (data['fold'][:, fold] == swap)
            if hp.redundant_predictors:
                lph_matrix[idx, fold] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '.feather')['LPH']
            else:
                lph_matrix[idx, fold] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '_no_redundancies.feather')['LPH']
    df_cml['LPH'] = lph_matrix.mean(axis=1)
    idx = (data['fold'][:, fold] != 99) #exclude validation fold
    df_cml = df_cml[idx].reset_index(drop=True)
    es_cml = EvalSurv(df_cml.copy())
    print('Base survival CML: {:.13}'.format(es_cml.get_base_surv(1826)))
    return    

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
            if hp.redundant_predictors:
                df_cml['LPH'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '.feather')['LPH']
            else:
                df_cml['LPH'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '_no_redundancies.feather')['LPH']
    
            ################################################################################################
                                        
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
    if hp.redundant_predictors:
        np.savez(hp.results_dir + 'eval_vecs_' + hp.gender + '.npz', 
                 r2_vec_cox=r2_vec_cox, d_index_vec_cox=d_index_vec_cox, concordance_vec_cox=concordance_vec_cox, ibs_vec_cox=ibs_vec_cox, auc_vec_cox=auc_vec_cox, 
                 r2_vec_cml=r2_vec_cml, d_index_vec_cml=d_index_vec_cml, concordance_vec_cml=concordance_vec_cml, ibs_vec_cml=ibs_vec_cml, auc_vec_cml=auc_vec_cml)
    else:
        np.savez(hp.results_dir + 'eval_vecs_' + hp.gender + '_no_redundancies.npz', 
                 r2_vec_cox=r2_vec_cox, d_index_vec_cox=d_index_vec_cox, concordance_vec_cox=concordance_vec_cox, ibs_vec_cox=ibs_vec_cox, auc_vec_cox=auc_vec_cox, 
                 r2_vec_cml=r2_vec_cml, d_index_vec_cml=d_index_vec_cml, concordance_vec_cml=concordance_vec_cml, ibs_vec_cml=ibs_vec_cml, auc_vec_cml=auc_vec_cml)

    
if __name__ == '__main__':
    main()

