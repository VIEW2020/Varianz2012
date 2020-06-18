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
    # Load data
    df = feather.read_dataframe(hp.data_dir + 'Py_VARIANZ_2012_v3-1.feather')
    
    # Exclude
    df.dropna(subset=['end_fu_date'], inplace=True)
    df = df[~df['ph_loopdiuretics_prior_5yrs_3evts'].astype(bool)]
    
    df['en_prtsd_eth'] = df['en_prtsd_eth'].astype(int)
    df['out_broad_cvd_adm_date'] = pd.to_datetime(df['out_broad_cvd_adm_date'], format='%Y-%m-%d', errors='coerce')
    df['end_fu_date'] = pd.to_datetime(df['end_fu_date'], format='%Y-%m-%d', errors='coerce')

    # Time to event and binary event column
    df['EVENT_DATE'] = df[['out_broad_cvd_adm_date', 'end_fu_date']].min(axis=1)
    beginning = pd.to_datetime({'year':[2012], 'month':[12], 'day':[31]})[0]
    df['TIME'] = (df['EVENT_DATE'] - beginning).dt.days.astype(int)
    df['EVENT'] = df['out_broad_cvd'] | df['imp_fatal_cvd']
    
    # Keep all relevant columns
    # keep_cols = ['VSIMPLE_INDEX_MASTER', 'nhi_age', 'gender_code', 'en_prtsd_eth', 'en_nzdep_q', 
    # 'hx_vdr_diabetes', 'hx_af', 'ph_antiplatelets_prior_6mths', 'ph_anticoagulants_prior_6mths', 
    # 'ph_lipid_lowering_prior_6mths', 'ph_bp_lowering_prior_6mths', 'TIME', 'EVENT']
    
    # Keep all relevant columns (exclude history of medications and hospital events to reduce collinearity issues)
    keep_cols = ['VSIMPLE_INDEX_MASTER', 'nhi_age', 'gender_code', 'en_nzdep_q', 'en_prtsd_eth', 'TIME', 'EVENT']
    df = df[keep_cols]
    
    # Save
    df.reset_index(drop=True, inplace=True)
    df.to_feather(hp.data_pp_dir + 'Py_VARIANZ_2012_v3-1_pp.feather')

if __name__ == '__main__':
    main()


