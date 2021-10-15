'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import feather
import pandas as pd
import numpy as np
from hyperparameters import Hyperparameters

from pdb import set_trace as bp

def main():
    hp = Hyperparameters()
    
    # Load data
    #df = feather.read_dataframe(hp.data_dir + 'Py_VARIANZ_2012_v3-1.feather')
    df = pd.read_feather(hp.data_dir + 'Py_VARIANZ_2012_v3-1.feather')
    
    # Exclude
    df = df[~df['ph_loopdiuretics_prior_5yrs_3evts'].astype(bool)]
    df = df[~df['ph_antianginals_prior_5yrs_3evts' ].astype(bool)]
    df.dropna(subset=['end_fu_date'], inplace=True)
    
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

    # Descriptive statistics
    num_participants = len(df.index)
    print('Total participants: {}'.format(num_participants))
    num_males = len(df.loc[df['gender_code']].index)
    num_females = len(df.loc[~df['gender_code']].index)
    print('Men: {} ({:.1f}%)'.format(num_males, 100*num_males/num_participants))
    print('Women: {} ({:.1f}%)'.format(num_females, 100*num_females/num_participants))
    mean_age_males, std_age_males = df.loc[df['gender_code'], 'nhi_age'].mean(), df.loc[df['gender_code'], 'nhi_age'].std()
    mean_age_females, std_age_females = df.loc[~df['gender_code'], 'nhi_age'].mean(), df.loc[~df['gender_code'], 'nhi_age'].std()
    print('Age Men: {:.1f} ({:.1f})'.format(mean_age_males, std_age_males))
    print('Age Women: {:.1f} ({:.1f})'.format(mean_age_females, std_age_females))
    num_nze_males = (df.loc[df['gender_code'], 'en_prtsd_eth'] == 1).sum()
    num_nze_females = (df.loc[~df['gender_code'], 'en_prtsd_eth'] == 1).sum()
    print('NZE Men: {} ({:.1f}%)'.format(num_nze_males, 100*num_nze_males/num_males))
    print('NZE Women: {} ({:.1f}%)'.format(num_nze_females, 100*num_nze_females/num_females))
    num_maori_males = (df.loc[df['gender_code'], 'en_prtsd_eth'] == 2).sum()
    num_maori_females = (df.loc[~df['gender_code'], 'en_prtsd_eth'] == 2).sum()
    print('Maori Men: {} ({:.1f}%)'.format(num_maori_males, 100*num_maori_males/num_males))
    print('Maori Women: {} ({:.1f}%)'.format(num_maori_females, 100*num_maori_females/num_females))
    num_pacific_males = (df.loc[df['gender_code'], 'en_prtsd_eth'] == 3).sum()
    num_pacific_females = (df.loc[~df['gender_code'], 'en_prtsd_eth'] == 3).sum()
    print('Pacific Men: {} ({:.1f}%)'.format(num_pacific_males, 100*num_pacific_males/num_males))
    print('Pacific Women: {} ({:.1f}%)'.format(num_pacific_females, 100*num_pacific_females/num_females))
    num_indian_males = (df.loc[df['gender_code'], 'en_prtsd_eth'] == 43).sum()
    num_indian_females = (df.loc[~df['gender_code'], 'en_prtsd_eth'] == 43).sum()
    print('Indian Men: {} ({:.1f}%)'.format(num_indian_males, 100*num_indian_males/num_males))
    print('Indian Women: {} ({:.1f}%)'.format(num_indian_females, 100*num_indian_females/num_females))
    num_other_males = (df.loc[df['gender_code'], 'en_prtsd_eth'] == 9).sum()
    num_other_females = (df.loc[~df['gender_code'], 'en_prtsd_eth'] == 9).sum()
    print('Other Men: {} ({:.1f}%)'.format(num_other_males, 100*num_other_males/num_males))
    print('Other Women: {} ({:.1f}%)'.format(num_other_females, 100*num_other_females/num_females))
    num_dp1_males = (df.loc[df['gender_code'], 'en_nzdep_q'] == 1).sum()
    num_dp1_females = (df.loc[~df['gender_code'], 'en_nzdep_q'] == 1).sum()
    print('dp1 Men: {} ({:.1f}%)'.format(num_dp1_males, 100*num_dp1_males/num_males))
    print('dp1 Women: {} ({:.1f}%)'.format(num_dp1_females, 100*num_dp1_females/num_females))
    num_dp2_males = (df.loc[df['gender_code'], 'en_nzdep_q'] == 2).sum()
    num_dp2_females = (df.loc[~df['gender_code'], 'en_nzdep_q'] == 2).sum()
    print('dp2 Men: {} ({:.1f}%)'.format(num_dp2_males, 100*num_dp2_males/num_males))
    print('dp2 Women: {} ({:.1f}%)'.format(num_dp2_females, 100*num_dp2_females/num_females))
    num_dp3_males = (df.loc[df['gender_code'], 'en_nzdep_q'] == 3).sum()
    num_dp3_females = (df.loc[~df['gender_code'], 'en_nzdep_q'] == 3).sum()
    print('dp3 Men: {} ({:.1f}%)'.format(num_dp3_males, 100*num_dp3_males/num_males))
    print('dp3 Women: {} ({:.1f}%)'.format(num_dp3_females, 100*num_dp3_females/num_females))
    num_dp4_males = (df.loc[df['gender_code'], 'en_nzdep_q'] == 4).sum()
    num_dp4_females = (df.loc[~df['gender_code'], 'en_nzdep_q'] == 4).sum()
    print('dp4 Men: {} ({:.1f}%)'.format(num_dp4_males, 100*num_dp4_males/num_males))
    print('dp4 Women: {} ({:.1f}%)'.format(num_dp4_females, 100*num_dp4_females/num_females))
    num_dp5_males = (df.loc[df['gender_code'], 'en_nzdep_q'] == 5).sum()
    num_dp5_females = (df.loc[~df['gender_code'], 'en_nzdep_q'] == 5).sum()
    print('dp5 Men: {} ({:.1f}%)'.format(num_dp5_males, 100*num_dp5_males/num_males))
    print('dp5 Women: {} ({:.1f}%)'.format(num_dp5_females, 100*num_dp5_females/num_females))
    num_diabetes_males = df.loc[df['gender_code'], 'hx_vdr_diabetes'].sum()
    num_diabetes_females = df.loc[~df['gender_code'], 'hx_vdr_diabetes'].sum()
    print('Diabetes Men: {} ({:.1f}%)'.format(num_diabetes_males, 100*num_diabetes_males/num_males))
    print('Diabetes Women: {} ({:.1f}%)'.format(num_diabetes_females, 100*num_diabetes_females/num_females))    
    num_AF_males = df.loc[df['gender_code'], 'hx_af'].sum()
    num_AF_females = df.loc[~df['gender_code'], 'hx_af'].sum()
    print('AF Men: {} ({:.1f}%)'.format(num_AF_males, 100*num_AF_males/num_males))
    print('AF Women: {} ({:.1f}%)'.format(num_AF_females, 100*num_AF_females/num_females))    
    num_BP_males = df.loc[df['gender_code'], 'ph_bp_lowering_prior_6mths'].sum()
    num_BP_females = df.loc[~df['gender_code'], 'ph_bp_lowering_prior_6mths'].sum()
    print('BP Men: {} ({:.1f}%)'.format(num_BP_males, 100*num_BP_males/num_males))
    print('BP Women: {} ({:.1f}%)'.format(num_BP_females, 100*num_BP_females/num_females))    
    num_LL_males = df.loc[df['gender_code'], 'ph_lipid_lowering_prior_6mths'].sum()
    num_LL_females = df.loc[~df['gender_code'], 'ph_lipid_lowering_prior_6mths'].sum()
    print('LL Men: {} ({:.1f}%)'.format(num_LL_males, 100*num_LL_males/num_males))
    print('LL Women: {} ({:.1f}%)'.format(num_LL_females, 100*num_LL_females/num_females))    
    num_APAC_males = df.loc[df['gender_code'], 'ph_antiplat_anticoag_prior_6mths'].sum()
    num_APAC_females = df.loc[~df['gender_code'], 'ph_antiplat_anticoag_prior_6mths'].sum()
    print('APAC Men: {} ({:.1f}%)'.format(num_APAC_males, 100*num_APAC_males/num_males))
    print('APAC Women: {} ({:.1f}%)'.format(num_APAC_females, 100*num_APAC_females/num_females))    
    follow_up_males, follow_up_males_mean = df.loc[df['gender_code'], 'TIME'].sum()/365, df.loc[df['gender_code'], 'TIME'].mean()/365
    follow_up_females, follow_up_females_mean = df.loc[~df['gender_code'], 'TIME'].sum()/365, df.loc[~df['gender_code'], 'TIME'].mean()/365
    print('Follow up Men: {:.0f} ({:.1f})'.format(follow_up_males, follow_up_males_mean))
    print('Follow up Women: {:.0f} ({:.1f})'.format(follow_up_females, follow_up_females_mean))    
    num_CVD_death_males = df.loc[df['gender_code'], 'imp_fatal_cvd'].sum()
    num_CVD_death_females = df.loc[~df['gender_code'], 'imp_fatal_cvd'].sum()
    print('CVD death Men: {} ({:.1f}%)'.format(num_CVD_death_males, 100*num_CVD_death_males/num_males))
    print('CVD death Women: {} ({:.1f}%)'.format(num_CVD_death_females, 100*num_CVD_death_females/num_females))    
    num_CVD_event_males = df.loc[df['gender_code'], 'EVENT'].sum()
    num_CVD_event_females = df.loc[~df['gender_code'], 'EVENT'].sum()
    print('CVD event Men: {} ({:.1f}%)'.format(num_CVD_event_males, 100*num_CVD_event_males/num_males))
    print('CVD event Women: {} ({:.1f}%)'.format(num_CVD_event_females, 100*num_CVD_event_females/num_females))
    tmp_males = df.loc[df['gender_code'] & df['EVENT'], 'TIME']/365
    time_to_CVD_males, time_to_CVD_males_Q1, time_to_CVD_males_Q3 = tmp_males.median(), tmp_males.quantile(0.25), tmp_males.quantile(0.75)
    tmp_females = df.loc[~df['gender_code'] & df['EVENT'], 'TIME']/365
    time_to_CVD_females, time_to_CVD_females_Q1, time_to_CVD_females_Q3 = tmp_females.median(), tmp_females.quantile(0.25), tmp_females.quantile(0.75)
    print('Time to CVD Men: {:.1f} ({:.1f}, {:.1f})'.format(time_to_CVD_males, time_to_CVD_males_Q1, time_to_CVD_males_Q3))
    print('Time to CVD Women: {:.1f} ({:.1f}, {:.1f})'.format(time_to_CVD_females, time_to_CVD_females_Q1, time_to_CVD_females_Q3))
    num_censored_5y_males = (1-df.loc[df['gender_code'] & (df['TIME'] == 1826), 'EVENT']).sum()
    num_censored_5y_females = (1-df.loc[~df['gender_code'] & (df['TIME'] == 1826), 'EVENT']).sum()
    print('Censored at 5 years Men: {} ({:.1f}%)'.format(num_censored_5y_males, 100*num_censored_5y_males/num_males))
    print('Censored at 5 years Women: {} ({:.1f}%)'.format(num_censored_5y_females, 100*num_censored_5y_females/num_females))

    # Center age and deprivation index, separately for males and females
    mean_age_males = df.loc[df['gender_code'], 'nhi_age'].mean()
    mean_age_females = df.loc[~df['gender_code'], 'nhi_age'].mean()
    df.loc[df['gender_code'], 'nhi_age'] =  df.loc[df['gender_code'], 'nhi_age'] - mean_age_males
    df.loc[~df['gender_code'], 'nhi_age'] =  df.loc[~df['gender_code'], 'nhi_age'] - mean_age_females
    
    mean_nzdep_males = 3
    mean_nzdep_females = 3
    df.loc[df['gender_code'], 'en_nzdep_q'] =  df.loc[df['gender_code'], 'en_nzdep_q'] - mean_nzdep_males
    df.loc[~df['gender_code'], 'en_nzdep_q'] =  df.loc[~df['gender_code'], 'en_nzdep_q'] - mean_nzdep_females
        
    # Create interaction columns
    df['age_X_bp'] = df['nhi_age'] * df['ph_bp_lowering_prior_6mths']
    df['age_X_diabetes'] = df['nhi_age'] * df['hx_vdr_diabetes']
    df['age_X_af'] = df['nhi_age'] * df['hx_af']
    df['bp_X_diabetes'] = df['ph_bp_lowering_prior_6mths'] & df['hx_vdr_diabetes']
    df['antiplat_anticoag_X_diabetes'] = df['ph_antiplat_anticoag_prior_6mths'] & df['hx_vdr_diabetes']
    df['bp_X_lipid'] = df['ph_bp_lowering_prior_6mths'] & df['ph_lipid_lowering_prior_6mths']
    
    # Keep all VARIANZ risk equations columns
    keep_cols = ['VSIMPLE_INDEX_MASTER', 'nhi_age', 'gender_code', 'en_prtsd_eth', 'en_nzdep_q', 
    'hx_vdr_diabetes', 'hx_af', 'ph_bp_lowering_prior_6mths', 'ph_lipid_lowering_prior_6mths',
    'ph_antiplat_anticoag_prior_6mths', 'age_X_bp', 'age_X_diabetes', 'age_X_af',
    'bp_X_diabetes', 'antiplat_anticoag_X_diabetes', 'bp_X_lipid', 'TIME', 'EVENT']
    df = df[keep_cols]
    
    # Save
    df_males = df[df['gender_code']]
    df_males.reset_index(drop=True, inplace=True)
    df_males.to_feather(hp.data_pp_dir + 'Py_VARIANZ_2012_v3-1_pp_males.feather')
    np.savez(hp.data_pp_dir + 'means_males.npz', mean_age=mean_age_males, mean_nzdep=mean_nzdep_males)

    df_females = df[~df['gender_code']]
    df_females.reset_index(drop=True, inplace=True)
    df_females.to_feather(hp.data_pp_dir + 'Py_VARIANZ_2012_v3-1_pp_females.feather')
    np.savez(hp.data_pp_dir + 'means_females.npz', mean_age=mean_age_females, mean_nzdep=mean_nzdep_females)


if __name__ == '__main__':
    main()


