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
    df = feather.read_dataframe(hp.data_dir + 'ALL_PHARMS_2008_2012_v3-1.feather')
    df['chem_id'] = df['chem_id'].astype(int)
    df['dispmonth_index'] = df['dispmonth_index'].astype(int)

    df.drop_duplicates(inplace=True)

    print('Remove future data...')
    df = df[df['dispmonth_index'] < 60]
    
    print('Invert time...')
    df['dispmonth_index'] = 59 - df['dispmonth_index']

    print('Remove codes associated with less than min_count persons...')
    df = df[df.groupby('chem_id')['VSIMPLE_INDEX_MASTER'].transform('nunique') >= hp.min_count]
    
    print('Save...')
    df.sort_values(by=['VSIMPLE_INDEX_MASTER', 'dispmonth_index', 'chem_id'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_feather(hp.data_pp_dir + 'PH_pp.feather')

if __name__ == '__main__':
    main()
    
