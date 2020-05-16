'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import feather
import pandas as pd
import numpy as np
import pickle as pkl
from hyperparameters import Hyperparameters as hp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from pycox.models import CoxTime
from tqdm import tqdm

from pdb import set_trace as bp


def sort_and_case_indices(x, time, event):
    """
    Sort data to allow for efficient sampling of people at risk.
    Time is in descending order, in case of ties non-events come first.
    In general, after sorting, if the index of A is smaller than the index of B,
    A is at risk when B experiences the event.
    To avoid sampling from ties, the column 'MAX_IDX_CONTROL' indicates the maximum
    index from which a case can be sampled.
    
    Args:
        x: input data
        time: time to event/censoring
        event: binary vector, 1 if the person experienced an event or 0 if censored
        
    Returns:
        sort_index: index to sort indices according to risk
        case_index: index to extract cases (on data sorted by sort_index!)
        max_idx_control: maximum index to sample a control for each case
    """
    # Sort
    df = pd.DataFrame({'TIME': time, 'EVENT': event.astype(bool)})
    df.sort_values(by=['TIME', 'EVENT'], ascending=[False, True], inplace=True)
    sort_index = df.index
    df.reset_index(drop=True, inplace=True)

    # Max idx for sampling controls (either earlier times or same time but no event)
    df['MAX_IDX_CONTROL'] = -1
    max_idx_control = -1
    prev_time = df.at[0, 'TIME']
    print('Computing MAX_IDX_CONTROL, time for a(nother) coffee...')
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not row['EVENT']:
            max_idx_control = i
        elif (prev_time > row['TIME']):
            max_idx_control = i-1
        df.at[i, 'MAX_IDX_CONTROL'] = max_idx_control
        prev_time = row['TIME']
    print('done')
    df_case = df[df['EVENT'] & (df['MAX_IDX_CONTROL']>=0)]
    case_index, max_idx_control = df_case.index, df_case['MAX_IDX_CONTROL'].values
    
    return sort_index, case_index, max_idx_control
  
  
def main():
    np.random.seed(hp.np_seed)

    print('Loading VARIANZ data...')
    df = feather.read_dataframe(hp.data_pp_dir + 'Py_VARIANZ_2012_v1_pp.feather')

    print('Loading medications...')
    ph = feather.read_dataframe(hp.data_pp_dir + 'PH_pp.feather')
    ph.rename(columns={'chem_id': 'CODE', 'dispmonth_index': 'MONTH'}, inplace=True)
    ph['TYPE'] = 0

    print('Loading hospital events...')
    he = feather.read_dataframe(hp.data_pp_dir + 'HE_pp.feather')
    he.rename(columns={'CLIN_CD_10': 'CODE', 'dispmonth_index': 'MONTH'}, inplace=True)
    he['TYPE'] = 1
    
    print('-----------------------------------------')

    # numerical index for each person
    df.reset_index(drop=True, inplace=True)
    df_index_person = df['VSIMPLE_INDEX_MASTER'].reset_index().rename(columns={'index': 'INDEX_PERSON'})

    # static variables
    print('Create dummy variables...')
    # Convert cathegorical ethnicity into indicator variables
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
    ac = ac.merge(df_index_person, how='left', on='VSIMPLE_INDEX_MASTER')
    print('Merging index_code...')
    ac['CODE'] = ac['CODE'].astype(str)
    ac = ac.merge(df_index_code,   how='left', on=['CODE', 'TYPE'])
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

    # Feature Transforms
    print('Scale continuous features...')
    cols_standardize = ['nhi_age']
    trans_list = []
    for col in df.columns.values.tolist():
        if col != 'TIME' and col !='EVENT' and col !='VSIMPLE_INDEX_MASTER':
            if col in cols_standardize:
                trans_list.append(([col], StandardScaler()))
            else:
                trans_list.append((col, None))
    x_mapper = DataFrameMapper(trans_list)
    
    x_trn = x_mapper.fit_transform(df_trn).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_tst = x_mapper.transform(df_tst).astype('float32')

    # Get target
    print('Get time and events...')
    get_target = lambda df: (df['TIME'].values, df['EVENT'].values)
    labtrans = CoxTime.label_transform()
    # Scale time for nonproportional hazards
    time_trn_nonprop, _ = labtrans.fit_transform(*get_target(df_trn)) # ignore event_trn_nonprop
    time_val_nonprop, _ = labtrans.transform(*get_target(df_val)) # ignore event_val_nonprop
    # For proportional hazards
    time_trn, event_trn = get_target(df_trn)
    time_val, event_val = get_target(df_val)
    time_tst, event_tst = get_target(df_tst)

    # Create datasets
    sort_idx_trn, case_idx_trn, max_idx_control_trn = sort_and_case_indices(x_trn, time_trn, event_trn)
    x_trn, time_trn, event_trn = x_trn[sort_idx_trn], time_trn[sort_idx_trn], event_trn[sort_idx_trn]
    codes_trn, month_trn, diagt_trn = codes_trn[sort_idx_trn], month_trn[sort_idx_trn], diagt_trn[sort_idx_trn]
    time_trn_nonprop = time_trn_nonprop[sort_idx_trn]
    
    sort_idx_val, case_idx_val, max_idx_control_val = sort_and_case_indices(x_val, time_val, event_val)
    x_val, time_val, event_val = x_val[sort_idx_val], time_val[sort_idx_val], event_val[sort_idx_val]
    codes_val, month_val, diagt_val = codes_val[sort_idx_val], month_val[sort_idx_val], diagt_val[sort_idx_val]
    time_val_nonprop = time_val_nonprop[sort_idx_val]
    
    print('-----------------------------------------')
    print('Save...')
    np.savez(hp.data_pp_dir + 'data_arrays.npz', x_trn=x_trn, time_trn=time_trn, event_trn=event_trn,
        codes_trn=codes_trn, month_trn=month_trn, diagt_trn=diagt_trn,
        case_idx_trn=case_idx_trn, max_idx_control_trn=max_idx_control_trn,
        x_val=x_val, time_val=time_val, event_val=event_val,
        codes_val=codes_val, month_val=month_val, diagt_val=diagt_val,
        case_idx_val=case_idx_val, max_idx_control_val=max_idx_control_val,
        x_tst=x_tst, time_tst=time_tst, event_tst=event_tst, codes_tst=codes_tst, month_tst=month_tst, diagt_tst=diagt_tst,)
    np.savez(hp.data_pp_dir + 'data_arrays_nonprop_hazards.npz', time_trn_nonprop=time_trn_nonprop, time_val_nonprop=time_val_nonprop)
    with open(hp.data_pp_dir + 'labtrans.pkl', 'wb') as f:
        pkl.dump(labtrans, f)
    df_index_code.to_feather(hp.data_pp_dir + 'df_index_code.feather')


if __name__ == '__main__':
    main()


