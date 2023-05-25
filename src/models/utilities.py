import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.utils import to_categorical


def change_dir(new_directory, directory):
    #Example of a new directory
    #new_directory = "C:/Users/ishaer/Documents/Occupancy Work/VTT_SCOTT_IAQ_dataset/VTT_SCOTT_IAQ_dataset/room00_train/combination"
    
    split_directory = directory.split("/")
    
    parent_directory = "/".join(split_directory[-2:])
    
    
    new_root_dir = new_directory + "/" + parent_directory
    
    
    return new_root_dir

#Temporarily changed that
def produce_train_validation_test_generators(set_models, model_set, preprocess_function):
        """
            This function produces train, test, and validation generators
        """

        df_train, df_validation, df_test = set_models['train'], set_models['validation'], set_models['test'] 
        predictor = model_set['predictor']
        output_variable = model_set['output_variable']

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_function) 
        train_generator = data_generator.flow_from_dataframe(dataframe = df_train, directory = None,
                                                    x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 32, seed = 13)

        validation_generator = data_generator.flow_from_dataframe(dataframe = df_validation, directory = None,
        x_col=predictor, y_col=output_variable, target_size = (96, 96),
        color_mode = "rgb", class_mode = "raw", 
        batch_size = 32, seed=13)

        testing_generator = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 32, seed = 13)

        return train_generator, validation_generator, testing_generator

def produce_train_test_generators_sets(set_models, model_set, preprocess_function):
    """
            This function produces train, test, and validation generators
        """

    df_train, df_test = set_models['train'], set_models['test'] 
    predictor = model_set['predictor']
    output_variable = model_set['output_variable']
    aux_variable = model_set['aux_variable']
    prediction_variable = model_set['prediction_variable']

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_function) 
    train_generator = data_generator.flow_from_dataframe(dataframe = df_train, directory = None,
                                                x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                color_mode = "rgb", class_mode = "raw", 
                                                batch_size = 16, seed = 13)
    train_generator_aux = data_generator.flow_from_dataframe(dataframe = df_train, directory = None,
                                                x_col=predictor, y_col=aux_variable, target_size = (96, 96),
                                                color_mode = "rgb", class_mode = "raw", 
                                                batch_size = 16, seed = 13)

    

    testing_generator = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                color_mode = "rgb", class_mode = "raw", 
                                                batch_size = 16, seed = 13)
    testing_generator_aux = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                x_col=predictor, y_col=aux_variable, target_size = (96, 96),
                                                color_mode = "rgb", class_mode = "raw", 
                                                batch_size = 16, seed = 13)

    testing_generator_prediction = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                x_col=predictor, y_col=prediction_variable, target_size = (96, 96),
                                                color_mode = "rgb", class_mode = "raw", 
                                                batch_size = 16, seed = 13)

    return train_generator, testing_generator, train_generator_aux, testing_generator_aux, testing_generator_prediction

def produce_train_validation_test_generators_sets(set_models, model_set, preprocess_function):
        """
            This function produces train, test, and validation generators
        """

        df_train, df_validation, df_test = set_models['train'], set_models['validation'], set_models['test'] 
        predictor = model_set['predictor']
        output_variable = model_set['output_variable']
        aux_variable = model_set['aux_variable']
        prediction_variable = model_set['prediction_variable']

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_function) 
        train_generator = data_generator.flow_from_dataframe(dataframe = df_train, directory = None,
                                                    x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)
        train_generator_aux = data_generator.flow_from_dataframe(dataframe = df_train, directory = None,
                                                    x_col=predictor, y_col=aux_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)

        validation_generator = data_generator.flow_from_dataframe(dataframe = df_validation, directory = None,
        x_col=predictor, y_col=output_variable, target_size = (96, 96),
        color_mode = "rgb", class_mode = "raw", 
        batch_size = 16, seed=13)

        validation_generator_aux = data_generator.flow_from_dataframe(dataframe = df_validation, directory = None,
                                                    x_col=predictor, y_col=aux_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)

        train_generator_prediction = data_generator.flow_from_dataframe(dataframe = df_train, directory = None,
                                                    x_col=predictor, y_col=prediction_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)

        validation_generator_prediction = data_generator.flow_from_dataframe(dataframe = df_validation, directory = None,
                                                    x_col=predictor, y_col=prediction_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)

        testing_generator = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)
        testing_generator_aux = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=aux_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)

        testing_generator_prediction = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=prediction_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)

        return train_generator, validation_generator, testing_generator, train_generator_aux, validation_generator_aux, testing_generator_aux, train_generator_prediction, validation_generator_prediction, testing_generator_prediction

def produce_feature_sets(set_models, model_set, preprocess_function):
        """
            This function produces train, test, and validation generators
        """

        df_train, df_validation, df_test = set_models['train'], set_models['validation'], set_models['test'] 
        predictor = model_set['predictor']
        output_variable = model_set['output_variable']
        aux_variables = model_set['aux_variables']
        prediction_variable = model_set['prediction_variable']

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_function) 
        train_generator = data_generator.flow_from_dataframe(dataframe = df_train, directory = None,
                                                    x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)
        

        validation_generator = data_generator.flow_from_dataframe(dataframe = df_validation, directory = None,
        x_col=predictor, y_col=output_variable, target_size = (96, 96),
        color_mode = "rgb", class_mode = "raw", 
        batch_size = 16, seed=13)


        validation_generator_prediction = data_generator.flow_from_dataframe(dataframe = df_validation, directory = None,
                                                    x_col=predictor, y_col=prediction_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)

        testing_generator = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)
        

        testing_generator_prediction = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=prediction_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)

        validation_generators_aux, testing_generators_aux = [], []

        for aux_variable in aux_variables:
            validation_generator_aux = data_generator.flow_from_dataframe(dataframe = df_validation, directory = None,
                                                    x_col=predictor, y_col=aux_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)
            testing_generator_aux = data_generator.flow_from_dataframe(dataframe = df_test, directory = None,
                                                    x_col=predictor, y_col=aux_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 16, seed = 13)
            validation_generators_aux.append(validation_generator_aux)
            testing_generators_aux.append(testing_generator_aux)


        return train_generator, validation_generator, testing_generator, validation_generator_prediction, testing_generator_prediction,validation_generators_aux,testing_generators_aux


def produce_better_generator(set_images, model_set, co2_values, preprocess_function):
    predictor = model_set['predictor']
    output_variable = model_set['output_variable']

    while True:
        np.random.seed(13)
        idx = np.random.permutation(set_images.shape[0])

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_function) 
        train_generator = data_generator.flow_from_dataframe(dataframe = set_images.iloc[idx], directory = None,
                                                    x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                    color_mode = "rgb", class_mode = "raw", 
                                                    batch_size = 32, random_state = 13)

        idx0 = 0
        for batch in train_generator:
            idx1 = idx0 + batch[0].shape[0]

            yield [batch[0], co2_values.iloc[ idx[ idx0:idx1 ] ]], batch[1]

            idx0 = idx1
            if idx1 >= set_images.shape[0]:
                break


def produce_three_generators(set_images, model_set, co2_values, preprocess_function):
    predictor = model_set['predictor']
    output_variable = model_set['output_variable']

    # while True:
    np.random.seed(13)
    idx = np.random.permutation(set_images.shape[0])

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_function) 
    train_generator = data_generator.flow_from_dataframe(dataframe = set_images.iloc[idx], directory = None,
                                                x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                color_mode = "rgb", class_mode = "raw", 
                                                batch_size = 32, random_state = 13)

    idx0 = 0
    for batch in train_generator:
        idx1 = idx0 + batch[0].shape[0]

        # yield [batch[0], co2_values.iloc[ idx[ idx0:idx1 ] ]], batch[1]
        yield batch[0], batch[1], co2_values.iloc[ idx[ idx0:idx1 ] ]

        idx0 = idx1
        if idx1 >= set_images.shape[0]:
            break

def produce_statement_generators(set_images, model_set, co2_values, preprocess_function):
    predictor = model_set['predictor']
    output_variable = model_set['output_variable']

    np.random.seed(13)
    idx = np.random.permutation(set_images.shape[0])

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_function) 
    train_generator = data_generator.flow_from_dataframe(dataframe = set_images.iloc[idx], directory = None,
                                                x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                color_mode = "rgb", class_mode = "raw", 
                                                batch_size = 32, random_state = 13)

    idx0 = 0
    for batch in train_generator:
        idx1 = idx0 + batch[0].shape[0]

        # yield [batch[0], co2_values.iloc[ idx[ idx0:idx1 ] ]], batch[1]
        yield batch[0], batch[1], co2_values.iloc[ idx[ idx0:idx1 ] ]

        idx0 = idx1
        if idx1 >= set_images.shape[0]:
            break

def produce_single_generator(set_images, model_set, co2_values, preprocess_function):
    predictor = model_set['predictor']
    output_variable = model_set['output_variable']

    np.random.seed(13)
    idx = np.random.permutation(set_images.shape[0])

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_function) 
    train_generator = data_generator.flow_from_dataframe(dataframe = set_images.iloc[idx], directory = None,
                                                x_col=predictor, y_col=output_variable, target_size = (96, 96),
                                                color_mode = "rgb", class_mode = "raw", 
                                                batch_size = 32, random_state = 13)

    idx0 = 0
    for batch in train_generator:
        idx1 = idx0 + batch[0].shape[0]

        yield [batch[0], co2_values.iloc[ idx[ idx0:idx1 ] ]], batch[1]

        idx0 = idx1
        if idx1 >= set_images.shape[0]:
            break

def load_base_models(target_size):
    resnet_101 = tf.keras.applications.resnet.ResNet101(include_top = False, weights = 'imagenet', input_shape=target_size)
    resnet_101.trainable =False
    resnet_152 = tf.keras.applications.resnet.ResNet152(include_top = False, weights = 'imagenet', input_shape=target_size)
    resnet_152.trainable = False
    resnet_50 = tf.keras.applications.resnet.ResNet50(include_top = False, weights = 'imagenet', input_shape=target_size)
    resnet_50.trainable = False
    xception = tf.keras.applications.Xception(include_top = False, weights = 'imagenet', input_shape=target_size)
    xception.trainable = False
    vgg_16 = tf.keras.applications.VGG16(include_top = False, weights = 'imagenet', input_shape=target_size)
    vgg_16.trainable = False
    vgg_19 = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet', input_shape=target_size)
    vgg_19.trainable = False
    inception_v3 = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet', input_shape=target_size)
    inception_v3.trainable = False
    inception_resnetV2 = tf.keras.applications.InceptionResNetV2(include_top = False, weights = 'imagenet', input_shape=target_size)
    inception_resnetV2.trainable = False

    set_base_models = {
        'resnet_101': resnet_101,
        'resnet_152': resnet_152,
        'resnet_50': resnet_50,
        'xception': xception,
        'vgg_16': vgg_16,
        'vgg_19': vgg_19,
        'inception_v3': inception_v3,
        'inception_resnetV2': inception_resnetV2,
    }

    return set_base_models

# Returns the name of the dense layer
def name_dense_layer(dense_layers):
    return "_".join([str(dense_layer) for dense_layer in dense_layers])

def reverse_dense_layer(name_layer):
    stripped_values = name_layer.split("_")
    if len(stripped_values) > 1:
        return [int(val) for val in stripped_values]
    else:
        if stripped_values[0] == '':
            return []
        else:
            return [int(stripped_values[0])]

# This function is responsible for returning the images found in a specific path
def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels = 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    return image

# Returns a generator of specific data
def retrieve_single_generator(data, batch_size = 64):
    data=tf.data.Dataset.from_tensor_slices(data)
    generator = data.batch(batch_size, drop_remainder=False)
    generator = generator.prefetch(8)

    return generator

# Returns a single generator that produces two types of data, in this case, the set of features and the set of labels
def retrieve_combined_generator(X_values, y_values, batch_size = 64):
    X_values=tf.data.Dataset.from_tensor_slices(X_values)
    X_values = X_values.map(preprocess_image)
    y_values = tf.data.Dataset.from_tensor_slices(y_values)

    generator = tf.data.Dataset.zip((X_values, y_values))
    generator = generator.batch(batch_size, drop_remainder=False)
    generator = generator.prefetch(8)

    return generator

'''
    This function produces the predictions for a set of features and returns the a metric defined by `fcn` parameter.
    model: the model that is evaluated.
    generator: the data generator that produces a set of features and set of labels
    fcn: the evaluation function
'''
def get_predictions(model, generator, fcn):
    set_predictions, set_values = [], []
    for idx, batch in enumerate(generator):
        validation_predictions = model.predict(batch[0])
        set_predictions.extend(np.argmax(validation_predictions, 1))
        set_values.extend(np.argmax(batch[1], 1))

    return fcn(set_values, set_predictions)

# Transform specific values into categorical
def encoding_retrieval(encoder, values):
    encoded_values = encoder.transform(values)
    return to_categorical(encoded_values)

def retrieve_labels(values):
    return np.concatenate([y for y in values], axis=0)


    