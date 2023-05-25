import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten, GlobalAveragePooling2D
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
import numpy as np

'''
    This code defines the function of Individual Learners
'''
class TunedCNNModel:

    def __init__(self, dense_layers, base_model):
        """
            - dense_layers: the array length represents the depth of the network and each value represents the number of units in each layer
            - base model: represents the feature extraction model used throughout this work
        """
        self.dense_layers = dense_layers
        self.base_model = base_model

    def create_CNN_model(self, list_params):
        inputs = keras.Input(shape=(156, 156, 3))
        kernel_size = list_params['kernel_size']
        dropout_value = list_params['dropout']


        x = Conv2D(list_params['filters'][0], kernel_size, padding="same", activation='relu')(inputs)
        x = BatchNormalization(axis = -1)(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)

        for i in range(1, len(list_params['filters'])):
            x = Conv2D(list_params['filters'][i], kernel_size, padding="same", activation='relu')(x)
            x = BatchNormalization(axis = -1)(x)
            x = MaxPooling2D(pool_size = (2, 2))(x)

        x = Flatten()(x)
        for (i, d) in enumerate(list_params["dense_layer"]):
            x = Dense(d, activation='relu')(x)
            x = Dropout(dropout_value)(x)
        output = Dense(1, activation='linear', name='output')(x)

        model = Model(inputs, output)
        model.compile(optimizer=Adam(learning_rate = list_params['lr']),
            loss={"output": 'mean_squared_error'},
            metrics = {'output': ["mean_absolute_error", "mean_absolute_percentage_error"]})
        
        self.model = model
        return model

    # def create_CNN_model(self, list_params):
    #     inputs = keras.Input(shape = (96, 96, 3))
    #     kernel_size = list_params['kernel_size']
    #     for (i, f) in enumerate(list_params['filters']):
	# 	# if this is the first CONV layer then set the input
	# 	# appropriately
    #         if i == 0:
    #             x = inputs
    #         # CONV => RELU => BN => POOL
    #         x = Conv2D(f, kernel_size, padding="same", activation='relu')(x)
    #         x = BatchNormalization(axis=-1)(x)
    #         x = MaxPooling2D(pool_size=(2, 2))(x)
    #     x = Flatten()(x)
    #     x = Dense(16, activation='relu')(x)
    #     x = BatchNormalization(axis=-1)(x)
    #     x = Dropout(0.5)(x)
    #     # apply another FC layer, this one to match the number of nodes
    #     # coming out of the MLP
    #     x = Dense(4,activation='relu')(x)
    #     # check to see if the regression node should be added
    #     x = Dense(1, activation="linear", name='output')(x)

    #     model = Model(inputs, x)
    #     model.compile(optimizer=Adam(learning_rate = list_params['lr']),
    #                 loss={"output": list_params['loss_fcn']},
    #                 metrics = {'output': ["mean_absolute_error", "mean_absolute_percentage_error"]})
    #     self.model = model
    #     return model

    def create_feature_extractor(self, list_params):
        inputs = Input(shape=(96, 96, 3))
        inter_model = self.base_model
        
        x = inter_model(inputs)
        global_max_layer = tf.keras.layers.GlobalMaxPooling2D()(x)

        model = Model(inputs, global_max_layer)

        model.compile(optimizer=Adam(learning_rate = list_params['lr']),
                loss={"output": list_params['loss_fcn']},
                metrics = {'output': ["mean_absolute_error", "mean_absolute_percentage_error"]})
        return model

    def create_regularized_model(self, list_params):
        inputs = keras.Input(shape=(96, 96, 3))
        # dense_layers = self.dense_layers
        inter_model = self.base_model
        
        x = inter_model(inputs)
        # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        global_average_layer = tf.keras.layers.GlobalMaxPooling2D()
        x = global_average_layer(x)
        x = BatchNormalization()(x)
        #512 is the outputsize
        output_size = x.shape[1]
        x = Dense(output_size)(x)
        output = Dense(units = 1, name='output', activity_regularizer = tf.keras.regularizers.L1(l1 = list_params['alpha']))(x)
        model = Model(inputs, output)
        model.compile(optimizer=Adam(learning_rate = list_params['lr']),
                    loss={"output": list_params['loss_fcn']},
                    metrics = {'output': ["mean_absolute_error", "mean_absolute_percentage_error"]})

        self.model = model
        return model

    def create_transferred_model(self, list_params):
        inputs = keras.Input(shape=(96, 96, 3))
        inter_model = self.base_model
        x = inter_model(inputs)
        x = Conv2D(32, 3, activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = GlobalAveragePooling2D()(x)

        final_output = Dense(units = 1, name='output')(x)

        model = Model(inputs, final_output)
        model.compile(optimizer=Adam(learning_rate = list_params['lr']),
                    loss={"output": list_params['loss_fcn']},
                    metrics = {'output': ["mean_absolute_error", "mean_absolute_percentage_error"]})

        return model

    def create_pca(self, n_c_variance):
        pca = PCA(n_c_variance)
        return pca

    def create_nn_model(self, input_shape, list_params):
        inputs = keras.Input(shape=(input_shape, ))
        dense_layers = self.dense_layers
        
        x = Dense(dense_layers[0], activation=list_params['activation_fcn'], name='tuned_layer_1')(inputs)
        if list_params['dropout'] > 0:
            x = Dropout(list_params['dropout'])(x)
        x = BatchNormalization()(x)
        if len(dense_layers) > 1:
            for idx, neurons in enumerate(dense_layers[1:]):
                x = Dense(units=neurons, activation=list_params['activation_fcn'], name='tuned_layer_{}'.format(str(idx+2)))(x)
                if list_params['dropout'] > 0:
                    x = Dropout(list_params['dropout'])(x)
                x = BatchNormalization()(x)
        output = Dense(units = 1, name='output')(x)
        model = Model(inputs, output)
        model.compile(optimizer=Adam(learning_rate = list_params['lr']),
                    loss={"output": list_params['loss_fcn']},
                    metrics = {'output': ["mean_absolute_error", "mean_absolute_percentage_error"]})
        
        self.model = model

        return model

    def create_model(self, list_params):
        """
            This method creates the model responsible for training image representations of time-series data 
            requires: defining the dense layers and the base model
            parameters: 
            - list_params: this parameters encompasses different hyper-parameters of the models except the dense layers. These hyper-parameters include:
                1) Dropout
                2) Loss function
                3) Learning Rate
                4) Activation Function
        """ 
        inputs = keras.Input(shape=(96, 96, 3))
        dense_layers = self.dense_layers
        inter_model = self.base_model
        
        x = inter_model(inputs)
        # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        global_average_layer = tf.keras.layers.GlobalMaxPooling2D()
        x = global_average_layer(x)
        # x = Dropout(list_params['dropout'])(x)
        # x = BatchNormalization()(x)
        x = Dense(dense_layers[0], activation=list_params['activation_fcn'], name='tuned_layer_1')(x)
        if len(dense_layers) > 1:
            for idx, neurons in enumerate(dense_layers[1:]):
                x = Dense(units=neurons, activation=list_params['activation_fcn'], name='tuned_layer_{}'.format(str(idx+2)))(x)
                if list_params['dropout'] > 0:
                    x = Dropout(list_params['dropout'])(x)
        output = Dense(units = 1, name='output')(x)
        model = Model(inputs, output)
        model.compile(optimizer=Adam(learning_rate = list_params['lr']),
                    loss={"output": list_params['loss_fcn']},
                    metrics = {'output': ["mean_absolute_error", "mean_absolute_percentage_error"]})

        self.model = model
        return model

    def create_classification_model(self, list_params):
        """
            This method creates the model responsible for training image representations of time-series data 
            requires: defining the dense layers and the base model
            parameters: 
            - list_params: this parameters encompasses different hyper-parameters of the models except the dense layers. These hyper-parameters include:
                1) Dropout
                2) Loss function
                3) Learning Rate
                4) Activation Function
        """ 
        inputs = keras.Input(shape=(156, 156, 3))
        dense_layers = self.dense_layers
        inter_model = self.base_model
        
        x = inter_model(inputs, training=False)
        global_average_layer = tf.keras.layers.GlobalMaxPooling2D()
        x = global_average_layer(x)
        x = Dropout(list_params['dropout'])(x)
        
        # x = BatchNormalization()(x)
        if len(dense_layers) > 0:
            x = Dense(dense_layers[0], activation=list_params['activation_fcn'], name='tuned_layer_1')(x)
        if len(dense_layers) > 1:
            for idx, neurons in enumerate(dense_layers[1:]):
                x = Dense(units=neurons, activation=list_params['activation_fcn'], name='tuned_layer_{}'.format(str(idx+2)))(x)
                if list_params['dropout'] > 0:
                    x = Dropout(list_params['dropout'])(x)
        output = Dense(units = 3, activation='sigmoid', name='output')(x)
        model = Model(inputs, output)
        loss_fn = keras.losses.CategoricalCrossentropy()
        model.compile(optimizer=Adam(learning_rate = list_params['lr']),
                    loss={"output": loss_fn},
                    metrics = {'output': ['accuracy', keras.metrics.AUC()]})

        self.model = model
        return model


    