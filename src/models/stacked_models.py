from asyncio.windows_utils import pipe
from email import generator
from subprocess import call
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import math
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.layers import LeakyReLU
from visualization.plotting_helpers import plot_predictions

"""
    This code represents the Ensemble Models
"""

class StackedDNNModels:


    def __init__(self, dense_layers, members,members_pca = None):
        self.dense_layers = dense_layers
        self.members = members

    def create_model(self, list_params):
        inputs = keras.Input(shape=(len(self.members), ))

        dense_layers = self.dense_layers
        x = Dense(dense_layers[0], activation=list_params['activation_fcn'], name='tuned_layer_1')(inputs)
        if len(dense_layers) > 1:
            for idx, neurons in enumerate(dense_layers[1:]):
                x = Dense(units=neurons, activation=list_params['activation_fcn'], name='tuned_layer_{}'.format(str(idx+2)))(x)
                if list_params['dropout'] > 0:
                    x = Dropout(list_params['dropout'])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output = Dense(units = 1, activation = list_params['activation_fcn'], name='output')(x)
        model = Model(inputs, output)
        model.compile(optimizer=Adam(learning_rate = list_params['lr']),
                    loss={"output": list_params['loss_fcn']},
                    metrics = {'output': ["mean_absolute_error", "mean_absolute_percentage_error"]})
        
        self.model = model

        return model
    
    def retrieve_labels(self, values):
        return np.concatenate([y for y in values], axis=0)

    def stacked_dataset(self, inputX):
        """
            This method produces the outputs of each individual model
        """
        stackX = None
        for idx, model in enumerate(self.members):
            data_input = inputX[idx]
            yhat = model.predict(data_input)
            yhat = yhat[:, 0] #Probability that the co2_var will decrease
            if stackX is None:
                stackX = yhat
            else:
                # stackX = np.hstack([stackX, yhat])
                stackX = np.column_stack([stackX, yhat])

        return stackX

    def fit_stacked_model(self, generators, labels, aux_generators, algorithm, filename):
        stackedX = self.stacked_dataset(generators)
        aux_variables = aux_generators[0]
        # for aux_generator in aux_generators:
        for i in range(1, len(aux_generators)):
            aux_variables = np.column_stack([aux_variables, aux_generators[i]])

       
        stackedX = np.column_stack([stackedX, aux_variables])
    
        model = None

        if algorithm == 'lr':
            scaler = StandardScaler()
            pipeline = Pipeline([("scaler",scaler),("model",LinearRegression())])
            model = pipeline
        elif algorithm == 'rf':
            model = RandomForestRegressor()
        elif algorithm == 'dt':
            model = DecisionTreeRegressor(max_depth=10)
        elif algorithm == 'ridge':
            scaler = StandardScaler()
            pipeline = Pipeline([("scaler",scaler),("model",Ridge())])
            model = pipeline
        else: 
            inputs = keras.Input(shape=(len(self.members)+ len(aux_generators), ))

            _ = Dense(units = 128, activation='relu')(inputs)
            _ = Dense(units = 64, activation='relu')(_)
            output = Dense(1, activation='relu', name='output')(_)

            model = Model(inputs = inputs, outputs = output)
            model.compile(optimizer=Adam(learning_rate = 1e-4),
                        loss={"output": 'mean_squared_error'},
                        metrics = {'output': ["mean_absolute_error", "mean_absolute_percentage_error"]})
        
        if algorithm != 'nn':
            model.fit(stackedX, labels)
        else:
            model.fit(stackedX, labels, epochs=100, verbose=0)
        predictions = model.predict(stackedX)
        self.model = model

        mae = mean_absolute_error(labels, predictions)
        plot_predictions(predictions, labels, filename, "MAE = {}".format(str(round(mae, 2))))

        return mean_absolute_error(labels, predictions), mean_absolute_percentage_error(labels, predictions)
    
    def ensemble_prediction(self, model, generators, labels, aux_generators):
        stackedX = self.stacked_dataset(generators)
        aux_variables = aux_generators[0]
        for i in range(1, len(aux_generators)):
            aux_variables = np.column_stack([aux_variables, aux_generators[i]])

        
        stackedX = np.column_stack([stackedX, aux_variables])

        predictions = model.predict(stackedX)
        # plot_predictions(predictions, labels, filename)

        # return mean_absolute_error(labels, predictions), mean_absolute_percentage_error(labels, predictions)
        return labels, predictions
        
        