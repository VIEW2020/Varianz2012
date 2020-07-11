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
    
    print('Replace DIAG_TYP with numerical values...')
    df.rename(columns={'DIAG_TYP': 'DIAG_TYPE'}, inplace=True)
    df['DIAG_TYPE'] = df['DIAG_TYPE'].replace({'A': 1, 'B': 2, 'E': 3, 'O': 4})

    print('Codes that actually exist as primary...')
    primary_codes = df.loc[df['DIAG_TYPE'] == 1, 'CLIN_CD_10'].drop_duplicates().reset_index()
    primary_codes.to_feather(hp.data_pp_dir + 'primary_codes.feather')

    print('Split males and females...')
    males = feather.read_dataframe(hp.data_pp_dir + 'Py_VARIANZ_2012_v3-1_pp_males.feather')['VSIMPLE_INDEX_MASTER']
    females = feather.read_dataframe(hp.data_pp_dir + 'Py_VARIANZ_2012_v3-1_pp_females.feather')['VSIMPLE_INDEX_MASTER']
    df_males = df.merge(males, how='inner', on='VSIMPLE_INDEX_MASTER')
    df_females = df.merge(females, how='inner', on='VSIMPLE_INDEX_MASTER')

    print('Remove codes associated with less than min_count persons...')
    df_males = df_males[df_males.groupby('CLIN_CD_10')['VSIMPLE_INDEX_MASTER'].transform('nunique') >= hp.min_count]
    df_females = df_females[df_females.groupby('CLIN_CD_10')['VSIMPLE_INDEX_MASTER'].transform('nunique') >= hp.min_count]
    
    print('Save...')
    df_males.sort_values(by=['VSIMPLE_INDEX_MASTER', 'dispmonth_index', 'CLIN_CD_10'], ascending=True, inplace=True)
    df_males.reset_index(drop=True, inplace=True)
    df_males.to_feather(hp.data_pp_dir + 'HE_pp_males.feather')

    df_females.sort_values(by=['VSIMPLE_INDEX_MASTER', 'dispmonth_index', 'CLIN_CD_10'], ascending=True, inplace=True)
    df_females.reset_index(drop=True, inplace=True)
    df_females.to_feather(hp.data_pp_dir + 'HE_pp_females.feather')    
    

if __name__ == '__main__':
    main()
    
