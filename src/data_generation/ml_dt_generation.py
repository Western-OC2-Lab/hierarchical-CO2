import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count


class DtGeneration:

    def __init__(self, root_dir, img_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir


    def parallelized_metrics(self, df_room, idx, granularity, set_columns):
         # The metrics to gather
        df_granularity = pd.DataFrame(columns = set_columns)

        set_params = {}
        set_params['set_columns'], set_params['granularity'], set_params['idx'] = df_granularity.columns, granularity, idx
        df_room.index = range(0, df_room.shape[0])
        metric_df = self.produce_metric_df(df_room, set_params)
        df_granularity = pd.concat([df_granularity, metric_df], axis = 0)

        return df_granularity


    def produce_metric_dfs(self, df_room, logging=False): 
        """
        This function is responsible for producing the dataframes for the metrics to be used for prediction purpose
        It takes the room's dataframe as a parameter
        """
        
        root_dir = self.root_dir
        set_columns = ["idx", "pir_mean", "pir_median", "pir_iqr", "pir_val", "pir_var", "pir_pp",
        "co2_mean", "co2_median", "co2_iqr", "co2_val", "co2_var", "co2_start", "co2_diff", "co2_max", "co2_pp",
        "humidity_pp", "temperature_pp", "pressure_pp", "co2_pir_pp", "temperature_pir_pp"]
        for granularity in [5, 10, 15, 20]:
            df_granularity = pd.DataFrame(columns = set_columns)
            set_granularity = Parallel(n_jobs=int(35), verbose=10)(delayed(self.parallelized_metrics)(dt, idx, granularity, set_columns) for idx, dt in enumerate(df_room))

            for df in set_granularity:
                df_granularity = pd.concat([df_granularity, df])

            df_granularity.to_csv("{}/{}_overlap-1_metrics.csv".format(root_dir, str(granularity)), index=None)

    def produce_metric_df(self, df, set_params = {}):

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
        """
            This function retrieves the metrics from each dataframe corresponding to a specific room. 
        """
        if 'set_columns' not in set_params.keys() or 'granularity' not in set_params.keys() or 'idx' not in set_params.keys():
            raise Exception('Missing one of the parameters "set_columns", "granularity", or "idx"')
        
        set_columns, granularity, idx = set_params['set_columns'], set_params['granularity'], set_params['idx']
        df.loc[:, 'pir_cnt_comp'] = df.pir_cnt.map(mapping) 
        df.loc[:, 'co2_pir'] = (df.loc[:, 'co2'].values / (df.loc[:, 'pir_cnt_comp'].values + 1))
        df.loc[:, 'temperature_pir'] = (df.loc[:, 'temperature'].values / (df.loc[:, 'pir_cnt_comp'].values + 1))


        metrics_df = pd.DataFrame(columns = set_columns)
        end_index, start_index = -1, -1 #Changed

        while end_index <= df.index[-1]:
            #changed this
            #start_index = end_index
            #end_index = start_index + (granularity + 1)
            start_index += 1
            end_index = start_index + (granularity - 1)

            sample_values_pir = df.loc[start_index:end_index, 'pir_cnt']
            sample_values_co2 = df.loc[start_index:end_index, 'co2']
            sample_values_temperature = df.loc[start_index:end_index, 'temperature']
            sample_values_humidity = df.loc[start_index:end_index, 'humidity']
            sample_values_pressure = df.loc[start_index:end_index, 'pressure']
            sample_values_co2_pir = df.loc[start_index:end_index, 'co2_pir']
            sample_values_temperature_pir = df.loc[start_index:end_index, 'temperature_pir']


            if sample_values_co2.shape[0] < granularity or sample_values_pir.shape[0] < granularity:
                break
            row_co2 = [round(np.mean(sample_values_co2), 2), 
                    round(np.median(sample_values_co2), 2), 
                    round(np.quantile(sample_values_co2, 0.75) - np.quantile(sample_values_co2, 0.25)),
                    sample_values_co2.iloc[granularity-1],
                    round( sample_values_co2.iloc[granularity-1] - sample_values_co2.iloc[0], 2 ),
                    sample_values_co2.iloc[0],
                    sample_values_co2.iloc[0]-np.max(sample_values_co2),
                    np.max(sample_values_co2),
                    np.max(sample_values_co2)-np.min(sample_values_co2)] 

            row_pir = [round(np.mean(sample_values_pir), 2), 
                    round(np.median(sample_values_pir), 2), 
                    round(np.quantile(sample_values_pir, 0.75) - np.quantile(sample_values_pir, 0.25)),
                    sample_values_pir.iloc[granularity-1],
                    round( sample_values_pir.iloc[granularity-1] - sample_values_pir.iloc[0], 2 ),
                    np.max(sample_values_pir)-np.min(sample_values_pir)
                    ]
              
            feature_stats = [
                np.max(sample_values_humidity)-np.min(sample_values_humidity),
                np.max(sample_values_temperature)-np.min(sample_values_temperature),
                np.max(sample_values_pressure)-np.min(sample_values_pressure),
                np.max(sample_values_co2_pir)-np.min(sample_values_co2_pir),
                np.max(sample_values_temperature_pir)-np.min(sample_values_temperature_pir),
            ]
           
            row = np.concatenate(([idx], row_pir, row_co2, feature_stats))

            metrics_df.loc[len(metrics_df)] = row
        metrics_df.index = range(metrics_df.shape[0])
        return metrics_df

    def produce_sets(self, df_idx, set_params={}):
        """
        This function returns the dataset that maps the combinatorial and individual features to the different metrics depending on the granularity
        """ 
        root_dir = self.root_dir
        if 'set_columns' not in set_params.keys() or 'granularity' not in set_params.keys() or 'idx' not in set_params.keys():
            raise Exception('Missing one of the parameters "set_columns", "granularity", or "idx"')
        
        set_columns, granularity, idx = set_params['set_columns'], set_params['granularity'], set_params['idx']
        set_name_prev_metrics = ["co2_h_start", "co2_h_var", "co2_h_pp", "humidity_h_pp", "temperature_h_pp", 
        "pir_h_pp", "pressure_h_pp", "co2_pir_h_pp", "temperature_pir_h_pp"]

        set_columns = list(set_columns)
        if set_name_prev_metrics[0] not in set_columns:
            set_columns.extend(set_name_prev_metrics)
        pd_h_f = pd.DataFrame(columns = set_columns)
        run = granularity

        while run < df_idx.shape[0]:
            set_metrics = ['pir_mean', 'pir_median', 'pir_iqr', "pir_val", "pir_var", "pir_pp", 'co2_mean', 'co2_median', 'co2_iqr', "co2_val", "co2_var", "co2_start", "co2_diff", "co2_max", "co2_pp"]
            set_prev_metrics = ["co2_start", "co2_var", "co2_pp", "humidity_pp", "temperature_pp", "pir_pp", "pressure_pp", "co2_pir_pp", "temperature_pir_pp"]
            metrics_row = df_idx.loc[run, set_metrics]

            prev_metrics = df_idx.loc[(run-granularity), set_prev_metrics]
            row = []
            set_individual = ['temperature', 'pressure', 'pir_cnt', 'co2', 'humidity']
            set_combinations = ["co2_pir_cnt_temperature_humidity", "co2_pir_cnt_temperature_pressure",
            "co2_temperature_humidity_pressure", "pir_cnt_temperature_humidity_pressure", "pir_cnt_co2_humidity_pressure"]

            file_name = "{root_dir}/{granularity}_{feature}/{idx}_{feature}_{run}.png"
        
            for indiv_feature in set_individual:
                app_file_name = file_name.format(root_dir=self.img_dir, granularity = granularity, feature=indiv_feature,
                                                idx = str(int(idx)), run = str(run-granularity))
                row.append(app_file_name)
                
            for combination in set_combinations:
                app_file_name = file_name.format(root_dir=self.img_dir, granularity = granularity, feature=combination,
                                                idx = str(int(idx)), run = str(run-granularity))
                row.append(app_file_name)

            row.extend(metrics_row)
            row.extend(prev_metrics)
            pd_h_f.loc[len(pd_h_f)] = row
            run += 1
        
        return pd_h_f

    def produce_inter_test(self, gran_df, idx, h):
        pd_h_f = pd.DataFrame(columns = ['temperature', 'pressure', 'pir_cnt', 'co2', 'humidity', 'c1', 'c2', 'c3', 'c4', 'c5',
                                        'pir_mean', 'pir_median', 'pir_iqr', "pir_val", "pir_var", "pir_pp", 'co2_mean', 'co2_median', 'co2_iqr', "co2_val", "co2_var", "co2_start", "co2_diff", "co2_max", "co2_pp"])
        df_idx = gran_df.loc[gran_df.idx == idx, :]
        df_idx.index = range(0, df_idx.shape[0])
        
        set_params = {}
        set_params['set_columns'], set_params['granularity'], set_params['idx']  = pd_h_f.columns, h, idx
        
        pd_idx = self.produce_sets(df_idx, set_params)
        pd_h_f = pd.concat([pd_h_f, pd_idx], axis = 0)

        return pd_h_f

    def produce_train_test_dfs(self, src_dir, dest_dir):
        """
            This function produces the sets to use for training and testing CNNs
        """
        for combination in [[5, 5], [10, 10], [15, 15], [20, 20]]:
            h, f = combination[0], combination[1]
            file_name = "h-{}_f-{}_overlap.csv".format(str(h), str(f))
            
            gran_df = pd.read_csv("{}/{}_overlap-1_metrics.csv".format(src_dir, str(f)))
            unique_idx = np.unique(gran_df.idx)
            
            pd_h_f = Parallel(n_jobs=int(35), verbose=10)(delayed(self.produce_inter_test)(gran_df, idx, h) for idx in unique_idx)
            new_df = pd.DataFrame(columns = pd_h_f[0].columns)

            for df in pd_h_f:
                new_df = pd.concat([new_df, df])
            # pd_h_f.index = range(0, pd_h_f.shape[0])
                
            new_df.to_csv("{}/{}".format(dest_dir, file_name))
            print("Done with {}".format(combination))