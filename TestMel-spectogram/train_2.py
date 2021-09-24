from __future__ import print_function
from __future__ import absolute_import

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D,Conv2D
from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Input, Dense

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os

from tensorflow.python.keras.saving.save import load_model

from keras_visualizer import visualizer


def build_model(input_shape,weights='msd', input_tensor=None,
                   include_top=True):

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    channel_axis = 1
    freq_axis = 2
    time_axis = 3
    # if K.image_dim_ordering() == 'th':
    #     channel_axis = 1
    #     freq_axis = 2
    #     time_axis = 3
    # else:
    #     channel_axis = 3
    #     freq_axis = 1
    #     time_axis = 2

    # Input block
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(melgram_input)

    # Conv block 1
    x = Convolution2D(64, 3, 3, padding='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis,  name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, padding='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis,  name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, padding='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis,  name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, padding='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis,  name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Conv block 5
    x = Convolution2D(64, 3, 3, padding='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis,  name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Output
    x = Flatten()(x)
    if include_top:
        x = Dense(9, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    return model


if __name__ == "__main__":

    model = keras.models.Sequential()
    model.add(Conv2D(8, (3, 3), padding="same",input_shape=(224,224,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(10))
    model.summary() 
    visualizer(model, format='png', view=True)

