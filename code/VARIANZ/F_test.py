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
import tqdm

from hyperparameters import Hyperparameters
from utils import *
import statsmodels.stats.api as sms

from pdb import set_trace as bp
    

def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    if hp.redundant_predictors:
        data = np.load(hp.results_dir + 'eval_vecs_' + hp.gender + '.npz')
    else:
        data = np.load(hp.results_dir + 'eval_vecs_' + hp.gender + '_no_redundancies.npz')

    # evaluation arrays
    r2_vec_cox = data['r2_vec_cox']
    d_index_vec_cox = data['d_index_vec_cox']
    concordance_vec_cox = data['concordance_vec_cox']
    ibs_vec_cox = data['ibs_vec_cox']
    auc_vec_cox = data['auc_vec_cox']

    r2_vec_cml = data['r2_vec_cml']
    d_index_vec_cml = data['d_index_vec_cml']
    concordance_vec_cml = data['concordance_vec_cml']
    ibs_vec_cml = data['ibs_vec_cml']
    auc_vec_cml = data['auc_vec_cml']

    r2_p = robust_cv_test(r2_vec_cox, r2_vec_cml)
    print('R-squared(D) p-value: {:.3}'.format(r2_p))
    d_index_p = robust_cv_test(d_index_vec_cox, d_index_vec_cml)
    print('D-index p-value: {:.3}'.format(d_index_p))
    concordance_p = robust_cv_test(concordance_vec_cox, concordance_vec_cml)
    print('Concordance p-value: {:.3}'.format(concordance_p))
    ibs_p = robust_cv_test(ibs_vec_cox, ibs_vec_cml)
    print('IBS p-value: {:.3}'.format(ibs_p))
    auc_p = robust_cv_test(auc_vec_cox, auc_vec_cml)
    print('AUC p-value: {:.3}'.format(auc_p))   

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

