#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import resize
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical


def preprocess_data(X, Y):
    """that pre-processes the data for your model:
args:
    X: np.ndarr of shape (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number of data points
    Y: np.ndarr of shape (m,) containing the CIFAR 10 labels for X
Returns: X_p, Y_p
    X_p is a numpy.ndarray containing the preprocessed X
    Y_p is a numpy.ndarray containing the preprocessed Y"""
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        preprocessing_function=lambda x: preprocess_input(resize(x, (224, 224), method='bilinear'))
    )
    
    generator = datagen.flow(X, Y)
    
    return generator

if __name__ == "__main__":

    #load train/valid data/test data/preprocess

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    valid_size = int(.2 * X_train.shape[0])

    X_trainb = X_train[:-valid_size]
    Y_trainb = Y_train[:-valid_size]
    X_valid = X_train[-valid_size:]
    Y_valid = Y_train[-valid_size:]

    train_gen = preprocess_data(X_trainb, Y_trainb)
    valid_gen = preprocess_data(X_valid, Y_valid)
    test_gen = preprocess_data(X_test, Y_test)

    #get pretrained model
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    #freeze layers
    for layer in base_model.layers:
        layer.trainable = False

    #build new model on top of pretrain
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    gap = GlobalAveragePooling2D()(x)
    fc1 = Dense(512, activation='relu')(gap)
    dropout = layers.Dropout(0.5)(fc1)
    outputs = Dense(10, activation='softmax')(dropout)
    model = models.Model(inputs, outputs)

    #compile
    model.compile(
        optimizer=Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'] 
        )


    #fit model
    history = model.fit(
        train_gen,
        validation_data = valid_gen,
        epochs=5,

    )
    #unfreeze after initial fitting/training
    for layer in base_model.layers[-13:]:
        layer.trainable = True

    #recompile
    model.compile(
        optimizer=Adam(1e-6),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    #continue training
    history_2 = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=20,
        batch_size=128
    )

    model.save('cifar10.h5')
