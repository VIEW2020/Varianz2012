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
        df_year = feather.read_dataframe(hp.data_dir + 'PHH_' + str(year) + '.feather')
        df_year['chem_id'] = df_year['chem_id'].astype(int)
        df_year['dispmonth_index'] = df_year['dispmonth_index'].astype(int)
        df = df.append(df_year)

    df.drop_duplicates(inplace=True)

    print('Remove future data...')
    df = df[df['dispmonth_index'] < 60]

    print('Remove codes associated with less than min_count persons...')
    df = df[df.groupby('chem_id')['VSIMPLE_INDEX_MASTER'].transform('nunique') >= hp.min_count]
    
    print('Save...')
    df.sort_values(by=['VSIMPLE_INDEX_MASTER', 'dispmonth_index', 'chem_id'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_feather(hp.data_pp_dir + 'PH_pp.feather')

if __name__ == '__main__':
    main()
    
