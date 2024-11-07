#!/usr/bin/env python3
"""module for building inception network based on Going Deeper with Convolutions (2014)"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block

def inception_network():
    """builds the inception network as described in Going Deeper with Convolutions (2014):
    
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should use (ReLU)
Returns: the keras model"""

    input = K.Input(shape=(224, 224, 3))

    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=7,
                             strides=2,
                             activation='relu',
                             padding='same')(input)
    max_pool_1 = K.layers.MaxPooling2D(pool_size=3,
                                       strides=2,
                                       padding='same')(conv_1)

    conv_2 = K.layers.Conv2D(filters=64,
                             kernel_size=1,
                             strides=1,
                             activation='relu')(max_pool_1)
    conv_2b = K.layers.Conv2D(filters=192,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(conv_2)
    max_pool_2 = K.layers.MaxPooling2D(pool_size=3,
                                       strides=2,
                                       padding='same')(conv_2b)

    inception_3a = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])
    max_pool_3 = K.layers.MaxPooling2D(pool_size=3,
                                       strides=2,
                                       padding='same')(inception_3b)

    inception_4a = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])
    max_pool_4 = K.layers.MaxPooling2D(pool_size=3,
                                       strides=2,
                                       padding='same')(inception_4e)

    inception_5a = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])
    avg_pool_1 = K.layers.AveragePooling2D(pool_size=7)(inception_5b)

    dropout_layer = K.layers.Dropout(0.4)(avg_pool_1)

    output = K.layers.Dense(1000, activation='softmax')(dropout_layer)

    model = K.models.Model(inputs=input, outputs=output)

    return model
