'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import sys
sys.path.append('../lib/')

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
        print('Mark validation data...')
        df_trn, df_tst = train_test_split(df, test_size=0.1, train_size=0.8, shuffle=True, stratify=df['EVENT'])
        df['VALIDATION'] = True
        df.loc[df_trn.index, 'VALIDATION'] = False
        df.loc[df_tst.index, 'VALIDATION'] = False

        # data folds stratified by event
        print('Split data into folds...')
        df['EVENT'] = df['EVENT'].astype(bool)
        num_event = (~df['VALIDATION'] & df['EVENT']).sum()
        num_nonevent = (~df['VALIDATION'] & ~df['EVENT']).sum()
        folds_event = np.zeros(num_event)
        folds_nonevent = np.zeros(num_nonevent)
        num_fold_event = int(np.floor(num_event/hp.num_folds))
        num_fold_nonevent = int(np.floor(num_nonevent/hp.num_folds))
        for i in range(1, hp.num_folds):
            folds_event[(i*num_fold_event):((i+1)*num_fold_event)] = i
            folds_nonevent[(i*num_fold_nonevent):((i+1)*num_fold_nonevent)] = i
        np.random.shuffle(folds_event)
        np.random.shuffle(folds_nonevent)
        df['FOLD'] = 0
        df.loc[~df['VALIDATION'] & df['EVENT'], 'FOLD'] = folds_event
        df.loc[~df['VALIDATION'] & ~df['EVENT'], 'FOLD'] = folds_nonevent
        df.loc[df['VALIDATION'], 'FOLD'] = 99 #don't consider for training/testing

        # Other arrays
        time = df['TIME'].values
        event = df['EVENT'].values.astype(int)
        fold = df['FOLD'].values
        df.drop(['TIME', 'EVENT', 'FOLD', 'VALIDATION', 'VSIMPLE_INDEX_MASTER', 'gender_code'], axis=1, inplace=True)
        x = df.values.astype('float32')
        
        print('-----------------------------------------')
        print('Save...')
        np.savez(hp.data_pp_dir + 'data_arrays_' + gender + '.npz', x=x, time=time, event=event, codes=codes, month=month, diagt=diagt, fold=fold)
        df_index_code.to_feather(hp.data_pp_dir + 'df_index_code_' + gender + '.feather')
        save_obj(list(df.columns), hp.data_pp_dir + 'cols_list.pkl')


if __name__ == '__main__':
    main()


