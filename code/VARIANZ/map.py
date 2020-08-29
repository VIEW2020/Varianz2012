'''
Aug 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import sys
sys.path.append('../lib/')

import feather
import pandas as pd
import numpy as np
import pickle as pkl
from hyperparameters import Hyperparameters

from tqdm import tqdm
from pandas_ods_reader import read_ods
from scipy.stats import pearsonr

from pdb import set_trace as bp
  

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

 
# From http://dept.stat.lsa.umich.edu/~kshedden/Python-Workshop/correlation_missing_data.html 
def corr(X, Y):
    """Computes the Pearson correlation coefficient and a 95% confidence
    interval based on the data in X and Y."""

    r = np.corrcoef(X, Y)[0,1]
    f = 0.5*np.log((1+r)/(1-r))
    se = 1/np.sqrt(len(X)-3)
    ucl = f + 2*se
    lcl = f - 2*se

    lcl = (np.exp(2*lcl) - 1) / (np.exp(2*lcl) + 1)
    ucl = (np.exp(2*ucl) - 1) / (np.exp(2*ucl) + 1)

    return r, lcl, ucl
 
 
def main():
    hp = Hyperparameters()

    for gender in ['females', 'males']:
        print(gender)
        data = np.load(hp.data_pp_dir + 'data_arrays_' + gender + '.npz')
        df = feather.read_dataframe(hp.data_pp_dir + 'df_index_person_' + gender + '.feather')
        df_geo = feather.read_dataframe(hp.data_dir + 'Py_VARIANZ_2012_v3-1_GEO.feather')[['VSIMPLE_INDEX_MASTER', 'MB2020_code']]
        df_mb_sa2 = read_ods(hp.data_dir + 'MB_SA2.ods', 1).rename(columns={'MB2020_V1_': 'MB2020_code'}).astype(int)
        df_geo = df_geo.merge(df_mb_sa2, how='left', on='MB2020_code').drop(['MB2020_code'], axis=1)
        df = df.merge(df_geo, how='left', on='VSIMPLE_INDEX_MASTER')
        
        # load predicted risk
        df['RISK_PERC'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + gender + '.feather')['RISK_PERC']

        # median risk
        print('Median risk: {:.3} IQR: [{:.3}, {:.3}]'.format(np.percentile(df['RISK_PERC'].values, 50), np.percentile(df['RISK_PERC'].values, 25), np.percentile(df['RISK_PERC'].values, 75))
        
        # set SA2s with less than 5 people to NaN
        df.loc[df.groupby('SA22020_V1')['VSIMPLE_INDEX_MASTER'].transform('nunique') < 5, 'RISK_PERC'] = np.nan
        
        # get median risk by SA2
        df = df.groupby('SA22020_V1').agg({'RISK_PERC': median}).reset_index()
        
        # save
        df.to_csv(hp.results_dir + 'df_sa2_' + gender + '.csv')
        if gender == 'females':
            df_females = df
        else:
            df_males = df

    bp()
    df = df_females.merge(df_males, on='SA2', how='inner')
    corr, _ = pearsonr(data1, data2)
    print('Pearsons correlation: {:.3}'.format(corr))
    corr, lcl, ucl = corr(data1, data2)
    print('Pearsons correlation: {:.3}'.format(corr))    
    

if __name__ == '__main__':
    main()


 
