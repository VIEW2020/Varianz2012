'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import sys
sys.path.append('../lib/')

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from EvalSurv import EvalSurv
from utils import *
import feather
from hyperparameters import Hyperparameters

from pdb import set_trace as bp


def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    
    print('Use all data for model fitting...')
    x = data['x']
    time = data['time']
    event = data['event']
    
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    df = pd.DataFrame(x, columns=cols_list)
    df['TIME'] = time
    df['EVENT'] = event

    ###################################################################
    
    print('Add additional columns...')
    df_index_code = pd.concat([df_index_code[df_index_code['TYPE']==1].head(10), df_index_code[df_index_code['TYPE']==0].head(10)], sort=False)
    for index, row in df_index_code.iterrows():
        df[row['DESCRIPTION']] = (data['codes'] == row['INDEX_CODE']).max(axis=1)
    
    ###################################################################
    
    print('Fitting...')
    cph = CoxPHFitter()
    cph.fit(df, duration_col='TIME', event_col='EVENT', show_progress=True, step_size=0.5)
    cph.print_summary()
    print('done')

if __name__ == '__main__':
    main()
