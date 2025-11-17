import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Input, BatchNormalization, Normalization
from keras import initializers
import numpy as np

def MLP_Model(x_train:np.ndarray, leaky_alpha:float = 0.5, first_layer_units:int = 32, second_layer_units:int = 20, third_layer_units:int = 12, activation:str = "sigmoid", dropout_rate:float = 0.5, use_norm_layer:bool = True)->Sequential: 
    
    """
    This function creates a Multi-Layer Perceptron (MLP) model using TensorFlow's Keras API. The model consists of three hidden layers with Leaky ReLU activation functions, dropout for regularization, and optional normalization of input features.
    The model is designed for binary classification tasks, with a single output neuron using the specified activation function (default is sigmoid) and was used for online movement prediction from combined ERP and PSD features.
    
    Parameters
    ----------
    x_train : np.ndarray
        The training data used to determine the input shape and for adapting the normalization layer if used. Should be in the shape (num_train_samples, num_features). 
    leaky_alpha : float, optional
        The hyperparameter of the LeakyReLU activation function, by default 0.5
    first_layer_units : int, optional
        The number of neurons in the first hidden layer, by default 32
    second_layer_units : int, optional
        The number of neurons in the second hidden layer, by default 20
    third_layer_units : int, optional
        The number of neurons in the third hidden layer, by default 12
    activation : str, optional
        The activation function of the output layer, by default "sigmoid"
    dropout_rate : float, optional
        The dropout rate used for regularization, by default 0.5
    use_norm_layer : bool, optional
        Flag if the feature normalization layer should be used, by default True

    Returns
    -------
    Sequential
        The sequential keras model representing the MLP.
    """
    
    # use He uniform initialization
    initializer = initializers.HeUniform()
    
    # MLP setup
    model = Sequential()
    model.add(Input(shape = x_train.shape[1],)) # calc input feature shape 
    
    # if add norm layer, use keras feature normalization layer
    if(use_norm_layer): 
        norm_layer = Normalization()
        norm_layer.adapt(x_train)
        model.add(norm_layer)
    
    model.add(Dense(units=first_layer_units, kernel_initializer = initializer))
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(units=second_layer_units, kernel_initializer = initializer))
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(units=third_layer_units, kernel_initializer = initializer)) 
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(Dense(units=1, activation=activation, kernel_initializer = initializer)) 
    
    return model 
 
    
