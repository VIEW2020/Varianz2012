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

from pdb import set_trace as bp
  

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

  
def main():
    hp = Hyperparameters()

    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    df = feather.read_dataframe(hp.data_pp_dir + 'df_index_person_' + hp.gender + '.feather')
    df_geo = feather.read_dataframe(hp.data_dir + 'Py_VARIANZ_2012_v3-1_GEO.feather')[['VSIMPLE_INDEX_MASTER', 'DHB_code', 'DHB_name']]
    df = df.merge(df_geo, how='left', on='VSIMPLE_INDEX_MASTER')
    
    # load predicted risk
    df['RISK_PERC'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '.feather')['RISK_PERC']
    
    # remove validation data
    df = df[data['fold'] != 99]
    
    # get median risk by DHB
    df = df.groupby('DHB_code').agg({'RISK_PERC': [percentile(50), percentile(80)], 'DHB_name': 'first'}).reset_index()
    
    # save
    df.to_csv(hp.results_dir + 'df_dhb_' + hp.gender + '.csv')


if __name__ == '__main__':
    main()


 
