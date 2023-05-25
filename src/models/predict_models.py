from matplotlib import testing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd
import time

from tensorflow import keras
import tensorflow as tf
from keras.models import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import load_model

class EvaluationDt: 

    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir

    # This function loads the dataset used for training data
    def get_model_sets(self): 
        data_dir = self.root_dir
        set_dfs = {
            'h-5_f-5': pd.read_csv("{}/model_sets/h-5_f-5_overlap.csv".format(data_dir), index_col=[0]),
            'h-10_f-10': pd.read_csv("{}/model_sets/h-10_f-10_overlap.csv".format(data_dir), index_col=[0]),
            'h-15_f-15': pd.read_csv("{}/model_sets/h-15_f-15_overlap.csv".format(data_dir), index_col=[0]),
            'h-20_f-20': pd.read_csv("{}/model_sets/h-20_f-20_overlap.csv".format(data_dir), index_col=[0])
        }
        set_features = ['co2', 'pir_cnt', 'pressure', 'temperature', 'humidity']
        set_h_f = list(set_dfs.keys())

        set_features_h = np.array(np.meshgrid(set_features,set_h_f)).T.reshape(-1, 2)

        return set_dfs, set_features_h

    def test_three_generators(self, df_test, model_set, preprocess_function):
        predictor = model_set['predictor']
        output_variable = model_set['output_variable']
        aux_variable = model_set['aux_variable']
        prediction_variable = model_set['prediction_variable']
        #No aggregation
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_function) 

        testing_generator = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 64)

        testing_generator_aux = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=aux_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 64)

        testing_generator_prediction = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=prediction_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 64)

        return testing_generator, testing_generator_aux, testing_generator_prediction

    def concat_models(self, base_model, loaded_model):
        inputs = keras.Input(shape=(96, 96, 3))
        inter_model = base_model # This variable represents the feature extractor

        x = inter_model(inputs)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        x = global_average_layer(x)


        for layer in loaded_model.layers[1:-1]:
            x = layer(x)
        output = loaded_model.layers[-1](x)
        model = Model(inputs, output)

        return model