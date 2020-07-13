'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import feather
import pandas as pd
import numpy as np
import pickle as pkl
from hyperparameters import Hyperparameters
from sklearn.model_selection import train_test_split
from deep_survival import sort_and_case_indices
from utils import save_obj

from tqdm import tqdm

from pdb import set_trace as bp
  
  
def main():
    hp = Hyperparameters()
    np.random.seed(hp.np_seed)

    for gender in ['males', 'females']:
        print('Processing ' + gender + '...')

        print('Loading VARIANZ data...')
        df = feather.read_dataframe(hp.data_pp_dir + 'Py_VARIANZ_2012_v3-1_pp_' + gender + '.feather')

        print('Loading medications...')
        ph = feather.read_dataframe(hp.data_pp_dir + 'PH_pp_' + gender + '.feather')
        ph.rename(columns={'chem_id': 'CODE', 'dispmonth_index': 'MONTH'}, inplace=True)
        ph['TYPE'] = 0

        print('Loading hospital events...')
        he = feather.read_dataframe(hp.data_pp_dir + 'HE_pp_' + gender + '.feather')
        he.rename(columns={'CLIN_CD_10': 'CODE', 'dispmonth_index': 'MONTH'}, inplace=True)
        he['TYPE'] = 1
        
        print('-----------------------------------------')

        # numerical index for each person
        df.reset_index(drop=True, inplace=True)
        df_index_person = df['VSIMPLE_INDEX_MASTER'].reset_index().rename(columns={'index': 'INDEX_PERSON'})

        # convert cathegorical ethnicity into indicator variables
        print('Create dummy variables...')
        df = pd.get_dummies(df, prefix='en_prtsd_eth', columns=['en_prtsd_eth'], drop_first=True)
        
        print('-----------------------------------------')
        print('Concatenating codes...')
        ac = pd.concat([ph, he], ignore_index=True, sort=False)
        ac['DIAG_TYPE'] = ac['DIAG_TYPE'].fillna(0).astype(int)

        # medications and hospital events
        print('Get max number of codes per person...')
        ac['COUNT'] = ac.groupby(['VSIMPLE_INDEX_MASTER']).cumcount()
        max_count = ac['COUNT'].max()+1
        print('max_count {}'.format(max_count))

        # code index (add 1 to reserve 0 for padding)
        df_index_code = ac[['CODE', 'TYPE']].drop_duplicates().reset_index(drop=True)
        df_index_code['CODE'] = df_index_code['CODE'].astype(str)
        df_index_code['INDEX_CODE'] = df_index_code.index + 1
            
        # codes, times, diag_type arrays
        codes = np.zeros((len(df_index_person), max_count), dtype=np.int16) # uint16 not supported by torch
        month = np.zeros((len(df_index_person), max_count), dtype=np.uint8)
        diagt = np.zeros((len(df_index_person), max_count), dtype=np.uint8)

        print('Merging index_person...')
        ac = ac.merge(df_index_person, how='inner', on='VSIMPLE_INDEX_MASTER')
        print('Merging index_code...')
        ac['CODE'] = ac['CODE'].astype(str)
        ac = ac.merge(df_index_code,   how='inner', on=['CODE', 'TYPE'])
        print('Updating arrays...')
        codes[ac['INDEX_PERSON'].values, ac['COUNT'].values] = ac['INDEX_CODE'].values
        month[ac['INDEX_PERSON'].values, ac['COUNT'].values] = ac['MONTH'].values
        diagt[ac['INDEX_PERSON'].values, ac['COUNT'].values] = ac['DIAG_TYPE'].values
        print('-----------------------------------------')

        # split data
        print('Split data into train/validate/test...')
        df_trn, df_tst = train_test_split(df, test_size=0.1, train_size=0.8, shuffle=True, stratify=df['EVENT'])
        df_val = df.drop(df_trn.index).drop(df_tst.index)
        
        codes_trn, codes_val, codes_tst = codes[df_trn.index], codes[df_val.index], codes[df_tst.index]
        month_trn, month_val, month_tst = month[df_trn.index], month[df_val.index], month[df_tst.index]
        diagt_trn, diagt_val, diagt_tst = diagt[df_trn.index], diagt[df_val.index], diagt[df_tst.index]

        # Create datasets
        sort_idx_trn, case_idx_trn, max_idx_control_trn = sort_and_case_indices(x_trn, time_trn, event_trn)
        x_trn, time_trn, event_trn = x_trn[sort_idx_trn], time_trn[sort_idx_trn], event_trn[sort_idx_trn]
        codes_trn, month_trn, diagt_trn = codes_trn[sort_idx_trn], month_trn[sort_idx_trn], diagt_trn[sort_idx_trn]
        
        sort_idx_val, case_idx_val, max_idx_control_val = sort_and_case_indices(x_val, time_val, event_val)
        x_val, time_val, event_val = x_val[sort_idx_val], time_val[sort_idx_val], event_val[sort_idx_val]
        codes_val, month_val, diagt_val = codes_val[sort_idx_val], month_val[sort_idx_val], diagt_val[sort_idx_val]
        
        print('-----------------------------------------')
        print('Save...')
        np.savez(hp.data_pp_dir + 'data_arrays_' + gender + '.npz', 
            x_trn=x_trn, time_trn=time_trn, event_trn=event_trn,
            codes_trn=codes_trn, month_trn=month_trn, diagt_trn=diagt_trn,
            case_idx_trn=case_idx_trn, max_idx_control_trn=max_idx_control_trn,
            x_val=x_val, time_val=time_val, event_val=event_val,
            codes_val=codes_val, month_val=month_val, diagt_val=diagt_val,
            case_idx_val=case_idx_val, max_idx_control_val=max_idx_control_val,
            x_tst=x_tst, time_tst=time_tst, event_tst=event_tst, 
            codes_tst=codes_tst, month_tst=month_tst, diagt_tst=diagt_tst)
        df_index_code.to_feather(hp.data_pp_dir + 'df_index_code_' + gender + '.feather')
        save_obj(cols_list, hp.data_pp_dir + 'cols_list.pkl')

if __name__ == '__main__':
    main()


