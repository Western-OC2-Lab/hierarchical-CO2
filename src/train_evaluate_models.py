from sklearn.model_selection import train_test_split
from models.individual_models import TunedCNNModel
from models.predict_models import EvaluationDt
from models.utilities import *
from models.stacked_models import StackedDNNModels
from sklearn.metrics import accuracy_score

from visualization.plotting_helpers import *
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')

class Train_Evaluate_Models:

    def __init__(self, h_f, root_dir, room_name, root_folder, runs):
        self.h_f = h_f
        self.target_size = (156, 156, 3)
        self.set_features = ['co2', 'co2_pir', 'temperature_pir', 'pressure', 'temperature', 'humidity']
        self.root_dir = root_dir
        self.room_name = room_name
        self.train_dir = f"{self.root_dir}/data/generated/{room_name}_train"
        self.root_folder = root_folder
        self.run = runs

        self.set_base_models = load_base_models(self.target_size)

        eval_dt = EvaluationDt(self.train_dir)
        set_dfs, _ = eval_dt.get_model_sets()
        self.all_dfs = set_dfs[self.h_f]
        self.all_dfs = self.change_df(self.all_dfs)
        

        # The hyper-parameters of the Individual Learners
        set_dense_layers = [[], [4096], [512], [256], [128], [64], [256, 128], [128, 64], [512, 256]]
        set_dense_layers = [name_dense_layer(layer) for layer in set_dense_layers]
        self.combinations = np.array(np.meshgrid(list(self.set_base_models.keys()), set_dense_layers)).T.reshape(-1, 2)
        self.validation_size, self.testing_size = 0.33, 0.1

        self.results = pd.DataFrame()
        self.encoder = LabelEncoder()
        self.encoder.fit(self.all_dfs.direction.values)

    def change_df(self, df):
        df = df.copy()
        # Defines the direction's categories
        def sign_values(x):
            if x < 0:
                return 0
            elif x == 0:
                return 1
            else:
                return 2
        
        df.loc[:, 'co2_pir'] = df.loc[:, 'co2'].apply(lambda x : x.replace("co2", "co2_pir"))
        df.loc[:, 'temperature_pir'] = df.loc[:, 'temperature'].apply(lambda x : x.replace("temperature", "temperature_pir"))
        df.loc[:, 'direction'] = 0

        df.direction = df.co2_var.apply(lambda x : sign_values(x)).astype('category')

        return df

    def train(self):
        self.create_insights_folders(
            self.root_dir,
            self.room_name,
            self.h_f,
            "direction_var_transformation"
        )
        # Defines the structure of the variable storing the hyper-parameters of individual models
        DENSE_LAYER, DROPOUT, LOSS_FCN, ACTIVATION, LEARNING_RATE = [64, 32], 0.2, 'mean_squared_error', 'relu', 1e-4
        hyper_params_indiv = {
            'dropout': DROPOUT,
            'loss_fcn': LOSS_FCN,
            'activation_fcn': ACTIVATION,
            'lr': LEARNING_RATE,
            'alpha': 0.1
        }

        # Defines the variables of the NN with Ensemble Model
        DENSE_LAYER_ENSEMBLE, DROPOUT_ENSEMBLE, LOSS_FCN_ENSEMBLE, ACTIVATION_ENSEMBLE, LEARNING_RATE_ENSEMBLE = [64], 0, 'mean_squared_error', 'relu', 1e-4
        hyper_params_ensemble = {
            'dropout': DROPOUT_ENSEMBLE,
            'loss_fcn': LOSS_FCN_ENSEMBLE,
            'activation_fcn': ACTIVATION_ENSEMBLE,
            'lr': LEARNING_RATE_ENSEMBLE
        }

        CALLBACK_INDIV = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta = 0.05, patience = 5, restore_best_weights = True)

        for run in range(self.run):
            random_state = 15 + run
            training_set, testing_set = train_test_split(self.sample_df, test_size=self.testing_size, random_state=random_state)
            training_set, validation_set = train_test_split(self.sample_df, test_size=self.validation_size, random_state = random_state)
            for combination in self.combinations:
                model_name = combination[0]
                dense_layer = combination[1]
                dict_results, dict_results['model_name'] = {}, model_name
                dict_results['nn'] = dense_layer
                dict_results['run'] = str(run)
                
                # These variables are saved so that the validation process is consistent
                # The `aux` variables refer to auxiliary variables. In this case, they represent the different between the start and the end of the history time window `e2s` in the paper and `h_pp` here
                validation_aux, testing_aux, set_models = [], [], []
                validation_generators, validation_generator_predictions = [], None
                testing_generators, testing_generator_predictions = [], None
                        
                for feature in self.set_features:
                    
                    print('Currently at Run-{} ModelName-{} DenseLayer-{} Feature-{}'.format(str(run), model_name, dense_layer, feature))
                    # We define the set of predictors and the response variable for the Individual Learners
                    X_train, y_train = training_set.loc[:, feature].values, training_set.loc[:, 'direction'].values
                    X_validation, y_validation = validation_set.loc[:, feature].values, validation_set.loc[:, 'direction'].values
                    X_test, y_test = testing_set.loc[:, feature].values, testing_set.loc[:, 'direction'].values

                    # We define the auxiliary variables that will be used for the ensemble model
                    feature_aux_validation, feature_aux_testing = validation_set.loc[:, "{}_h_pp".format(feature)].values,  testing_set.loc[:, "{}_h_pp".format(feature)].values
                    y_var_validation = validation_set.loc[:, 'co2_var'].values
                    validation_p = retrieve_single_generator(y_var_validation)
                    validation_generator_predictions = validation_p

                    y_var_testing = testing_set.loc[:, 'co2_var'].values
                    testing_p = retrieve_single_generator(y_var_testing)
                    testing_generator_predictions = testing_p

                    encoded_Y_train = self.encoder.transform(y_train)
                    encoded_Y_validation = self.encoder.transform(y_validation)

                    y_train, y_validation = encoding_retrieval(self.encoder, y_train), encoding_retrieval(self.encoder, y_validation)
                    validation_aux.append(feature_aux_validation)
                    testing_aux.append(feature_aux_testing)

                    base_model = self.set_base_models[model_name]
                    
                    cnn_model = TunedCNNModel(reverse_dense_layer(dense_layer), base_model)        
                    
                    tr_model = cnn_model.create_classification_model(hyper_params_indiv)

                    validation_generator, training_generator, testing_generator = retrieve_combined_generator(X_validation, y_validation), retrieve_combined_generator(X_train, y_train), retrieve_combined_generator(X_test, y_test)
                    validation_generators.append(validation_generator)
                    testing_generators.append(testing_generator)


                    start_time = time.time()

                    tr_model.fit(training_generator, validation_data = validation_generator, epochs = 15, verbose=0, callbacks=[CALLBACK_INDIV])
                    end_time = time.time()

                    training_time_feature = round((end_time - start_time) / 60, 2)
                    dict_results['training_time_{}'.format(feature)] = training_time_feature
                    set_models.append(tr_model)
                    
                    dict_results['acc_val_{}'.format(feature)] = get_predictions(tr_model, validation_generator, accuracy_score)

                    print('res - {} - {} - {}'.format(model_name, feature, dict_results['acc_val_{}'.format(feature)]))

                set_ensemble_models = []
                for algorithm in ['lr', 'nn', 'rf', 'dt', 'ridge']:
                    model_ensemble = StackedDNNModels(DENSE_LAYER_ENSEMBLE, set_models, None)
                    figures_filename = "{}/ValidationEnsemble-{}-{}-{}-{}.png".format(self.FIGURES_DIR, model_name, algorithm, dense_layer, str(run))
                    start_time = time.time()
                    dict_results['MAE_val_{}'.format(algorithm)], dict_results['MAPE_val_{}'.format(algorithm)] = model_ensemble.fit_stacked_model(validation_generators, retrieve_labels(validation_generator_predictions), validation_aux, algorithm, figures_filename)
                    end_time = time.time()
                    training_time_ensemble = round((end_time - start_time) / 60, 2)
                    dict_results['training_time_{}'.format(algorithm)] = training_time_ensemble

                    print('Ensemble validation MAE {} {} {}: {}'.format(model_name, algorithm, str(run), str(dict_results['MAE_val_{}'.format(algorithm)])))
                    set_ensemble_models.append(model_ensemble.model)

                    start_time = time.time()
                    labels, predictions = model_ensemble.ensemble_prediction(model_ensemble.model, testing_generators, retrieve_labels(testing_generator_predictions), testing_aux)
                    end_time = time.time()
                    testing_time_ensemble = round((end_time - start_time) / 60, 2)

                    dict_results['testing_time_{}'.format(algorithm)] = testing_time_ensemble


                    for threshold in [None, 5, 10, 20, 40, 50, 75, 100]:
                        figures_filename = "{}/TestingEnsemble-M{}-A{}-D{}-R{}-T{}.png".format(self.FIGURES_DIR, model_name, algorithm, dense_layer, str(run), str(threshold))
                        dict_results['MAE_test-{}-{}'.format(algorithm, str(threshold))] = defining_plotting_thresholds(predictions, labels, figures_filename, threshold_values = threshold)
                        print('Ensemble Test-Threshold -{}: {}'.format(str(threshold), str(dict_results['MAE_test-{}-{}'.format(algorithm, str(threshold))])))
                    
                self.results = pd.concat([self.results, pd.DataFrame.from_dict([dict_results])])
                self.results.to_csv("{}/M{}-D{}.csv".format(self.STATS_DIR, model_name, dense_layer))

    def create_insights_folders(self, root_folder, room_name, h_f, method, type_op = 'train'):
        FIGURES_DIR =  f"{root_folder}/reports/figures/{room_name}/{type_op}/{h_f}/{method}"
        STATS_DIR = f"{root_folder}/reports/results/{room_name}/{type_op}/{h_f}/{method}"
        MODELS_DIR = f"{root_folder}/models/{room_name}/{h_f}/{method}"

        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(STATS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        self.FIGURES_DIR = FIGURES_DIR
        self.STATS_DIR = STATS_DIR
        self.MODELS_DIR = MODELS_DIR


