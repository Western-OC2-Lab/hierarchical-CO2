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
import joblib
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from train_evaluate_models import Train_Evaluate_Models
import matplotlib
import pickle

matplotlib.use('Agg')

class Models_Generation(Train_Evaluate_Models):

    def __init__(self, h_f, root_dir, room_name, root_folder, runs):
        super(Models_Generation, self).__init__(h_f, root_dir, room_name, root_folder, runs)
        self.best_models = {
            'h-5_f-5': {
                'method': 'direction_var_transformation',
                'algorithm': 'dt',
                'model': 'xception',
                'nn': [512]
            },
            'h-10_f-10': {
                'method': 'direction_var_transformation',
                'algorithm': 'dt',
                'model': 'resnet_50',
                'nn': [128]
            },
            'h-15_f-15': {
                'method': 'direction_var_transformation',
                'algorithm': 'dt',
                'model': 'resnet_152',
                'nn': [256, 128]
            },
            'h-20_f-20': {
                'method': 'direction_var_transformation',
                'algorithm': 'dt',
                'model': 'resnet_101',
                'nn': [128, 64]
            },
            
        }

        self.test_dir = f"{self.root_dir}/data/generated/{room_name}_test"
        train_dt = EvaluationDt(self.train_dir)
        self.training_dfs, _ = train_dt.get_model_sets()

        test_dt = EvaluationDt(self.test_dir)
        self.testing_dfs, _ = test_dt.get_model_sets()

    def train(self):
        results = pd.DataFrame()
        validation_size = 0.33

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

        for h_f in list(self.best_models.keys()):
            params = self.best_models[h_f]
            method, algorithm, model_name, dense_layer = params['method'], params['algorithm'], params['model'], params['nn']

            training_df = self.training_dfs[h_f]
            testing_df = self.testing_dfs[h_f]

            training_df = self.change_df(training_df)
            testing_df = self.change_df(testing_df)

            self.create_insights_folders(self.root_dir, self.room_name, h_f, method, 'train')


            encoder = LabelEncoder()
            encoder.fit(training_df.direction.values)

            random_state = 15
            training_set, validation_set = train_test_split(training_df, test_size=validation_size, random_state = random_state)
            dict_results, dict_results['model_name'] = {}, model_name
            dict_results['nn'] = dense_layer
            dict_results['h_f'] = h_f
            
            validation_aux, testing_aux, set_models = [], [], []
            validation_generators, validation_generator_predictions = [], None
            testing_generators, testing_generator_predictions = [], None
                    
            for feature in self.set_features:
                
                print('Currently at ModelName-{} DenseLayer-{} Feature-{}'.format(model_name, name_dense_layer(dense_layer), feature))
                X_train, y_train = training_set.loc[:, feature].values, training_set.loc[:, 'direction'].values
                X_validation, y_validation = validation_set.loc[:, feature].values, validation_set.loc[:, 'direction'].values
                X_test, y_test = testing_df.loc[:, feature].values, testing_df.loc[:, 'direction'].values

                feature_aux_validation, feature_aux_testing = validation_set.loc[:, "{}_h_pp".format(feature)].values,  testing_df.loc[:, "{}_h_pp".format(feature)].values
                y_var_validation = validation_set.loc[:, 'co2_var'].values
                validation_p = retrieve_single_generator(y_var_validation)
                validation_generator_predictions = validation_p

                y_var_testing = testing_df.loc[:, 'co2_var'].values
                testing_p = retrieve_single_generator(y_var_testing)
                testing_generator_predictions = testing_p

                y_train, y_validation = encoding_retrieval(encoder, y_train), encoding_retrieval(encoder, y_validation)
                validation_aux.append(feature_aux_validation)
                testing_aux.append(feature_aux_testing)

                base_model = self.set_base_models[model_name]
                
                cnn_model = TunedCNNModel(dense_layer, base_model)        
                
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
                tr_model.save("{}/{}.h5".format(self.MODELS_DIR, feature))

                
                dict_results['acc_val_{}'.format(feature)] = get_predictions(tr_model, validation_generator, accuracy_score)

                print('res - {} - {} - {}'.format(model_name, feature, dict_results['acc_val_{}'.format(feature)]))

            set_ensemble_models = []
            model_ensemble = StackedDNNModels(DENSE_LAYER_ENSEMBLE, set_models, None)
            figures_filename = "{}/ValidationEnsemble-{}-{}-{}.png".format(self.FIGURES_DIR, model_name, algorithm, name_dense_layer(dense_layer))
            start_time = time.time()
            dict_results['MAE_val_{}'.format(algorithm)], dict_results['MAPE_val_{}'.format(algorithm)] = model_ensemble.fit_stacked_model(validation_generators, retrieve_labels(validation_generator_predictions), validation_aux, algorithm, figures_filename)
            end_time = time.time()
            training_time_ensemble = round((end_time - start_time) / 60, 2)
            dict_results['training_time_{}'.format(algorithm)] = training_time_ensemble

            print('Ensemble validation MAE {} {}: {}'.format(model_name, algorithm, str(dict_results['MAE_val_{}'.format(algorithm)])))
            set_ensemble_models.append(model_ensemble.model)

            start_time = time.time()
            labels, predictions = model_ensemble.ensemble_prediction(model_ensemble.model, testing_generators, retrieve_labels(testing_generator_predictions), testing_aux)
            end_time = time.time()
            testing_time_ensemble = round((end_time - start_time) / 60, 2)

            dict_results['testing_time_{}'.format(algorithm)] = testing_time_ensemble

            for threshold in [None, 5, 10, 20, 40, 50, 75, 100]:
                figures_filename = "{}/TestingEnsemble-M{}-A{}-D{}-T{}.png".format(self.FIGURES_DIR, model_name, algorithm, name_dense_layer(dense_layer), str(threshold))
                dict_results['MAE_test-{}-{}'.format(algorithm, str(threshold))] = defining_plotting_thresholds(predictions, labels, figures_filename, threshold_values = threshold)
                print('Ensemble Test-Threshold -{}: {}'.format(str(threshold), str(dict_results['MAE_test-{}-{}'.format(algorithm, str(threshold))])))
            
            filename = f'{algorithm}.sav'
            pickle.dump(model_ensemble.model, open("{}/{}".format(self.MODELS_DIR, filename), 'wb'))

            results = pd.concat([results, pd.DataFrame.from_dict([dict_results])])
            results.to_csv("{}/M{}-D{}-{}.csv".format(self.STATS_DIR, model_name, name_dense_layer(dense_layer), h_f))
        




    

