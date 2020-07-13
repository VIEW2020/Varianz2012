'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import feather
import pandas as pd
from hyperparameters import Hyperparameters

from pdb import set_trace as bp

def main():
    hp = Hyperparameters()
    
    # Load data
    df = feather.read_dataframe(hp.data_dir + 'Py_VARIANZ_2012_v3-1.feather')
    
    # Exclude
    df.dropna(subset=['end_fu_date'], inplace=True)
    df = df[~df['ph_loopdiuretics_prior_5yrs_3evts'].astype(bool)]
    df = df[~df['ph_antianginals_prior_5yrs_3evts' ].astype(bool)]
    
    # Adjust data types
    df['nhi_age'] = df['nhi_age'].astype(int)
    df['gender_code'] = df['gender_code'].astype(bool)
    df['en_prtsd_eth'] = df['en_prtsd_eth'].astype(int)
    df['en_nzdep_q'] = df['en_nzdep_q'].astype(int)
    df['hx_vdr_diabetes'] = df['hx_vdr_diabetes'].astype(bool)
    df['hx_af'] = df['hx_af'].astype(bool)
    df['ph_bp_lowering_prior_6mths'] = df['ph_bp_lowering_prior_6mths'].astype(bool)
    df['ph_lipid_lowering_prior_6mths'] = df['ph_lipid_lowering_prior_6mths'].astype(bool)
    df['ph_anticoagulants_prior_6mths'] = df['ph_anticoagulants_prior_6mths'].astype(bool)
    df['ph_antiplatelets_prior_6mths'] = df['ph_antiplatelets_prior_6mths'].astype(bool)
    df['out_broad_cvd_adm_date'] = pd.to_datetime(df['out_broad_cvd_adm_date'], format='%Y-%m-%d', errors='coerce')
    df['end_fu_date'] = pd.to_datetime(df['end_fu_date'], format='%Y-%m-%d', errors='coerce')

    # Map Other Asian, Chinese, MELAA to 'other'
    df['en_prtsd_eth'].replace({4:9, 42:9, 5:9}, inplace=True)
    
    # Create antiplatelet/anticoagulant column
    df['ph_antiplat_anticoag_prior_6mths'] = df['ph_antiplatelets_prior_6mths'] | df['ph_anticoagulants_prior_6mths']
    
    # Time to event and binary event column
    df['EVENT_DATE'] = df[['out_broad_cvd_adm_date', 'end_fu_date']].min(axis=1)
    beginning = pd.to_datetime({'year':[2012], 'month':[12], 'day':[31]})[0]
    df['TIME'] = (df['EVENT_DATE'] - beginning).dt.days.astype(int)
    df['EVENT'] = df['out_broad_cvd'] | df['imp_fatal_cvd']
    
    # Center age and deprivation index, separately for males and females
    mean_age_males = df.loc[df['gender_code'], 'nhi_age'].mean()
    mean_age_females = df.loc[~df['gender_code'], 'nhi_age'].mean()
    df.loc[df['gender_code'], 'nhi_age'] =  df.loc[df['gender_code'], 'nhi_age'] - mean_age_males
    df.loc[~df['gender_code'], 'nhi_age'] =  df.loc[~df['gender_code'], 'nhi_age'] - mean_age_females
    
    mean_nzdep_males = df.loc[df['gender_code'], 'en_nzdep_q'].mean()
    mean_nzdep_females = df.loc[~df['gender_code'], 'en_nzdep_q'].mean()    
    df.loc[df['gender_code'], 'en_nzdep_q'] =  df.loc[df['gender_code'], 'en_nzdep_q'] - mean_nzdep_males
    df.loc[~df['gender_code'], 'en_nzdep_q'] =  df.loc[~df['gender_code'], 'en_nzdep_q'] - mean_nzdep_females
    
    print('Mean age (males, females): {},{}'.format(mean_age_males, mean_age_females))
    print('Mean nzdep (males, females): {},{}'.format(mean_nzdep_males, mean_nzdep_females))
    
    # Create interaction columns
    df['age_X_bp'] = df['nhi_age'] * df['ph_bp_lowering_prior_6mths']
    df['age_X_diabetes'] = df['nhi_age'] * df['hx_vdr_diabetes']
    df['age_X_af'] = df['nhi_age'] * df['hx_af']
    df['bp_X_diabetes'] = df['ph_bp_lowering_prior_6mths'] & df['hx_vdr_diabetes']
    df['antiplat_anticoag_X_diabetes'] = df['ph_antiplat_anticoag_prior_6mths'] & df['hx_vdr_diabetes']
    df['bp_X_af'] = df['ph_bp_lowering_prior_6mths'] & df['hx_af']
    
    # Keep all VARIANZ risk equations columns
    keep_cols = ['VSIMPLE_INDEX_MASTER', 'nhi_age', 'gender_code', 'en_prtsd_eth', 'en_nzdep_q', 
    'hx_vdr_diabetes', 'hx_af', 'ph_bp_lowering_prior_6mths', 'ph_lipid_lowering_prior_6mths',
    'ph_antiplat_anticoag_prior_6mths', 'age_X_bp', 'age_X_diabetes', 'age_X_af',
    'bp_X_diabetes', 'antiplat_anticoag_X_diabetes', 'bp_X_af', 'TIME', 'EVENT']
    df = df[keep_cols]
    
    # Save
    df_males = df[df['gender_code']]
    df_males.reset_index(drop=True, inplace=True)
    df_males.to_feather(hp.data_pp_dir + 'Py_VARIANZ_2012_v3-1_pp_males.feather')

    df_females = df[~df['gender_code']]
    df_females.reset_index(drop=True, inplace=True)
    df_females.to_feather(hp.data_pp_dir + 'Py_VARIANZ_2012_v3-1_pp_females.feather')


if __name__ == '__main__':
    main()


