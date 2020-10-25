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
import pickle as pkl

from utils import *
from hyperparameters import Hyperparameters
import optuna
from tqdm import tqdm

from pdb import set_trace as bp


def main():
    pp = Hyperparameters()
    
    print('Load data...')
    data = np.load(pp.data_pp_dir + 'data_arrays_' + pp.gender + '.npz')
    df_index_code = feather.read_dataframe(pp.data_pp_dir + 'df_index_code_' + pp.gender + '.feather')

    codes = data['codes']
    code_cols = np.zeros((codes.shape[0], df_index_code.shape[0]), dtype=bool)

    print('Codes to columns...')
    for i in tqdm(range(df_index_code.shape[0])):
        code_cols[:, i] = np.bitwise_or.reduce(codes == (i+1), 1)
    df_code_cols = pd.DataFrame(code_cols, columns = [str(i+1) for i in range(df_index_code.shape[0])])
    
    print('Save...')
    df_code_cols.to_feather(pp.data_pp_dir + 'df_code_cols_' + pp.gender + '.feather')


if __name__ == '__main__':
    main()
