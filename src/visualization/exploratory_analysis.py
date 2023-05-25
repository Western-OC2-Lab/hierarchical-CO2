import pandas as pd
import numpy as np
import os
import pickle

def correlation_dfs(folder_dir, feature, future=True):
    set_cols = ["corr_{}".format(i) for i in range(60)]
    set_cols.append("dataset")
    corr_dfs = pd.DataFrame(columns=set_cols)

    n, name = 1, "future" 
    if future == False:
        n, name = -1, "past"

    for file in os.listdir("{}".format(folder_dir)):
        if os.path.isdir("{}/{}".format(folder_dir, file)) == False: 
            df = pd.read_pickle("{}/{}".format(folder_dir, file))
            
            list_corr = np.zeros(60)
            for i in range(60):
                list_corr[i] = df['co2'].corr(df[feature].shift(periods = (n) * (-i)))
            
            row = pd.DataFrame(columns = set_cols)
            row.loc[0, 'dataset'] = file
            row.loc[0, set_cols[:-1]] = list_corr
            
            corr_dfs = corr_dfs.append(row)

    corr_dfs.index = range(corr_dfs.shape[0])
    rng = np.arange(0.2, 1, 0.1)
    dt_idx = {}

    for idx in range(corr_dfs.shape[0]):
        dt = corr_dfs.loc[idx, 'dataset']
        
        dt_idx[dt] = {}
        set_corr = set_cols[:-1]
        
        dt_arr = np.array(corr_dfs.loc[idx, set_corr])
        dt_idx[dt]['highest_idx'] = np.argmax(dt_arr)
        dt_idx[dt]['highest_value'] = corr_dfs.loc[idx, "corr_{}".format(dt_idx[dt]['highest_idx'])]
        for val in rng:
            val_idx = -10
            set_idxes = np.where(dt_arr <= val)
            set_idxes = set_idxes[0]
            if set_idxes.size > 0: 
                val_idx = set_idxes[0]
            dt_idx[dt]['{}_idx'.format(round(val, 2))] = val_idx  

    dt_idx = pd.DataFrame.from_dict(dt_idx)
    dt_idx.dropna(inplace=True, axis = 1)

    with open("{}/best_corr_{}_{}.pickle".format(folder_dir, feature, name), "wb") as handle:
        pickle.dump(dt_idx, handle)

    with open("{}/corr_stats_{}_{}.pickle".format(folder_dir, feature, name), "wb") as handle:
        pickle.dump(corr_dfs, handle)