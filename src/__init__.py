from joblib import Parallel, delayed
import pandas as pd
import os

from data_generation.ts_image_transformation import TSTransformation


continuous_dir = "C:/Users/ishaer/Documents/Occupancy Work/VTT_SCOTT_IAQ_dataset/data_cleaning/cleaned_data/continuous_sections_60_train.pickle"
room_name = 'room01'

df_cont = pd.read_pickle(continuous_dir)
df_room = df_cont[room_name]

set_combinations = {
    "c1": ["co2", "pir_cnt", "temperature", "humidity"],
    "c2": ["co2", "pir_cnt", "temperature", "pressure"],
    "c3": ["co2", "temperature", "humidity", "pressure"],
    "c4": ["pir_cnt", "temperature", "humidity", "pressure"],
    'c5': ['pir_cnt', "co2", "humidity", "pressure"]
}

def generate_images(folder_dir, idx, df, feature, granularity):
    saving_params = {}
    saving_params['folder_dir'] = folder_dir
    saving_params['file_name'] = str(idx)

    tsTransform = TSTransformation(saving_params['folder_dir'], df, granularity)
    tsTransform.apply_gaf_single_feature(feature, saving_params)

    return saving_params

folder_dir = "../../data/room01_train/individual"
run = 0
set_granularities = [5, 10, 15, 20]
set_features =  ['co2', 'pir_cnt', 'temperature', 'humidity', 'pressure']
for granularity in set_granularities:
    for feature in set_features:
        n_folder_dir = "{}/{}_{}".format(folder_dir, str(granularity), feature)
        os.mkdir(n_folder_dir)

        values = Parallel(n_jobs=8, verbose=10)(delayed(generate_images)(n_folder_dir, idx, df, feature, granularity) for idx, df in enumerate(df_room))

    #     for idx, df in enumerate(df_room):

    #         saving_params = {}
    #         saving_params['folder_dir'] = n_folder_dir
    #         saving_params['file_name'] = str(idx)
            
    #         tsTransform = TSTransformation(n_folder_dir, df, granularity)
    #         # apply_gaf_single_feature(df, granularity, feature, saving_params)

    #         if run % 30 == 0 and run != 0:
    #             print('Current at {} out of {} for feature {} and granularity {}'.format(str(idx), str(len(df_room)), feature, str(granularity)))

    #         run += 1
    # print("Done for {} and {}".format(str(granularity), feature))

# values = Parallel(n_jobs=8, verbose=10)(delayed()() for granularity in set_granularities for feature in set_features)