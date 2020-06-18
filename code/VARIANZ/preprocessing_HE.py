'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import feather
import pandas as pd
from hyperparameters import Hyperparameters as hp

from pdb import set_trace as bp


def main():
    df = feather.read_dataframe(hp.data_dir + 'HX_ADM_2008_2012_v3-1.feather')
    df.rename(columns={'eventmonth_index': 'dispmonth_index'}, inplace=True)
    df['dispmonth_index'] = df['dispmonth_index'].astype(int)

    df.drop_duplicates(inplace=True)

    print('Remove future data...')
    df = df[df['dispmonth_index'] < 60]
    
    print('Invert time...')
    df['dispmonth_index'] = 59 - df['dispmonth_index']

    print('Remove codes associated with less than min_count persons...')
    df = df[df.groupby('CLIN_CD_10')['VSIMPLE_INDEX_MASTER'].transform('nunique') >= hp.min_count]
    
    print('Replace DIAG_TYP with numerical values...')
    df.rename(columns={'DIAG_TYP': 'DIAG_TYPE'}, inplace=True)
    df['DIAG_TYPE'] = df['DIAG_TYPE'].replace({'A': 0, 'B': 1, 'E': 2, 'O': 3})

    print('Codes that actually exist as primary...')
    primary_codes = df.loc[df['DIAG_TYPE'] == 0, 'CLIN_CD_10'].drop_duplicates().reset_index()
    primary_codes.to_feather(hp.data_pp_dir + 'primary_codes.feather')

    print('Save...')
    df.sort_values(by=['VSIMPLE_INDEX_MASTER', 'dispmonth_index', 'CLIN_CD_10'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_feather(hp.data_pp_dir + 'HE_pp.feather')

if __name__ == '__main__':
    main()
    
