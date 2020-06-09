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
    df = pd.DataFrame()
    
    for year in range(2008, 2013):
        # Load data
        print('Year: {}'.format(year))
        df_year = feather.read_dataframe(hp.data_dir + 'HX_ADM_v2_' + str(year) + '.feather')
        df_year.rename(columns={'eventmonth_index': 'dispmonth_index'}, inplace=True)
        df_year['dispmonth_index'] = df_year['dispmonth_index'].astype(int)
        df = df.append(df_year)

    df.drop_duplicates(inplace=True)

    print('Remove future data...')
    df = df[df['dispmonth_index'] < 60]

    print('Remove codes associated with less than min_count persons...')
    df = df[df.groupby('CLIN_CD_10')['VSIMPLE_INDEX_MASTER'].transform('nunique') >= hp.min_count]
    
    print('Replace DIAG_TYP with numerical values...')
    df.rename(columns={'DIAG_TYP': 'DIAG_TYPE'}, inplace=True)
    df['DIAG_TYPE'] = df['DIAG_TYPE'].replace({'A': 0, 'B': 1, 'E': 2, 'O': 3})

    print('Codes that actually exist as primary...')
    primary_codes = df.loc[df['DIAG_TYPE'] == 1, 'CLIN_CD_10'].drop_duplicates().reset_index()
    primary_codes.to_feather(hp.data_pp_dir + 'primary_codes.feather')

    print('Save...')
    df.sort_values(by=['VSIMPLE_INDEX_MASTER', 'dispmonth_index', 'CLIN_CD_10'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_feather(hp.data_pp_dir + 'HE_pp.feather')

if __name__ == '__main__':
    main()
    
