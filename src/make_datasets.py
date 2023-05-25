from joblib import Parallel, delayed, cpu_count
import pandas as pd
import os
import numpy as np
import joblib

from data_generation.ts_image_transformation import TSTransformation
from data_generation.ml_dt_generation import DtGeneration


set_combinations = {
    "c1": ["co2", "pir_cnt", "temperature", "humidity"],
    "c2": ["co2", "pir_cnt", "temperature", "pressure"],
    "c3": ["co2", "temperature", "humidity", "pressure"],
    "c4": ["pir_cnt", "temperature", "humidity", "pressure"],
    'c5': ['pir_cnt', "co2", "humidity", "pressure"]
}

#Based on the transformed values 
#These values are extracted from the training set of room00 and should be applied to the testing set
SCALED_VALUES = {
    'temperature':  {'min': 6.98, 'max': 13.31},
    'humidity':  {'min': 1.06, 'max': 2.17},
    'pressure':  {'min': 15382.7, 'max': 31539.6},
    'co2':  {'min': 2.58, 'max': 3.43},
    'co2_pir':  {'min': 1.90, 'max': 3.38},
    'temperature_pir':  {'min': 0.70, 'max': 1.43},
}

mapping = {
        0: 0,
        
        0.5: 1,
        1: 1,
        1.5: 1,
        2: 1,
        2.5: 1,
        3: 1,
        3.5: 1,
        4: 1,
        
        4.5: 2,
        5: 2,
        5.5: 2,
        6: 2,
        6.5: 2,
        7: 2,
        7.5: 2,
        8: 2,
        
        8.5: 3,
        9: 3,
        9.5: 3,
        10: 3,
        10.5: 3,
        11: 3,
        11.5: 3,
        12:3
    }

def generate_images(folder_dir, idx, df, feature, granularity, type_generation):
    
    saving_params = {}
    saving_params['folder_dir'] = folder_dir
    saving_params['file_name'] = str(idx)

    df.loc[:, 'pir_cnt_comp'] = df.pir_cnt.map(mapping) 
    df.loc[:, 'co2_pir'] = (df.loc[:, 'co2'].values / (df.loc[:, 'pir_cnt_comp'].values + 1))
    df.loc[:, 'temperature_pir'] = (df.loc[:, 'temperature'].values / (df.loc[:, 'pir_cnt_comp'].values + 1))

    df.drop(columns = ['pir_cnt'], inplace=True)

    if feature == 'co2' or feature == 'humidity':
        df.loc[:, feature] = np.log10(df.loc[:, feature].values + 1)
    elif feature == 'temperature':
        df.loc[:, feature] = np.exp( df.loc[:, feature].values / 10)
    elif feature == 'pressure':
        df.loc[:, feature] = np.exp( df.loc[:, feature].values / 100)
    else:
        df.loc[:, feature] = np.log10( (df.loc[:, feature].values / (df.loc[:, 'pir_cnt_comp'].values + 1) +1) + 1)


    df.loc[:, feature] = (df.loc[:, feature] - SCALED_VALUES[feature]['min']) / (SCALED_VALUES[feature]['max'] - SCALED_VALUES[feature]['min'])
    tsTransform = TSTransformation(saving_params['folder_dir'], df, granularity)
    
    if type_generation == 'local':
        tsTransform.apply_gaf_single_feature(feature, saving_params)
    else:
        tsTransform.apply_edited_gaf_single_feature(feature, saving_params)

    return saving_params

# types_data can be train or test
for types_data in ['test']:
    room_name = "room01"

    continuous_dir = f"../data/raw/continuous_sections_60_{types_data}_{room_name}.pickle"


    df_cont = pd.read_pickle(continuous_dir)
    df_room = df_cont[room_name]

    folder_dir = "../data/generated/{}_{}/local".format(room_name, types_data)
    os.makedirs(folder_dir, exist_ok=True)
    set_granularities = [5, 10]

    set_features =  ['temperature', 'humidity', 'pressure', 'co2_pir', 'temperature_pir', 'co2']

    for granularity in set_granularities:
        for feature in set_features:
            n_folder_dir = "{}/{}_{}".format(folder_dir, str(granularity), feature)
            os.makedirs(n_folder_dir, exist_ok=True)

            values = Parallel(n_jobs=int(35), verbose=10)(delayed(generate_images)(n_folder_dir, idx, df, feature, granularity, 'local') for idx, df in enumerate(df_room))

    # root_dir = "../data/generated/{}_{}/model_sets".format(room_name, types_data)
    # os.makedirs(root_dir, exist_ok=True)
    
    # dtGen = DtGeneration(root_dir, img_dir=folder_dir)

    # dtGen.produce_metric_dfs(df_room, True)
    # dtGen.produce_train_test_dfs(src_dir=root_dir, dest_dir=root_dir)