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
  
  
def main():
    hp = Hyperparameters()

    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    df = feather.read_dataframe(hp.data_pp_dir + 'df_index_person_' + hp.gender + '.feather')[['VSIMPLE_INDEX_MASTER', 'DHB_code', 'DHB_name']]
    df_geo = feather.read_dataframe(hp.data_dir + 'Py_VARIANZ_2012_v3-1_GEO.feather')
    df = df.merge(df_geo, how='left', on='VSIMPLE_INDEX_MASTER')
    
    # load predicted risk
    df['RISK'] = 0
    for fold in range(hp.num_folds):
        df.loc[data['fold'] == fold, 'RISK'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '.feather')['ENSEMBLE'].values
    
    # remove validation data
    df = df[data['fold'] != 99]
    
    # get median risk by DHB
    bp()
    df = df.groupby('DHB_code').agg({'RISK': 'np.median', 'DHB_name': 'first'})
    

    


if __name__ == '__main__':
    main()


 
