#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.image import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical


def preprocess_data(X, Y):
    """that pre-processes the data for your model:
args:
    X: np.ndarr of shape (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number of data points
    Y: np.ndarr of shape (m,) containing the CIFAR 10 labels for X
Returns: X_p, Y_p
    X_p is a numpy.ndarray containing the preprocessed X
    Y_p is a numpy.ndarray containing the preprocessed Y"""
    
    # One-hot encode the labels
    Y_p = to_categorical(Y, 10)
    
    # Apply EfficientNet's preprocessing
    X_p = X.astype('float32')
    X_p = preprocess_input(X_p)
    
    return X_p, Y_p
if __name__ == "__main__":

    #load train/valid data/test data/preprocess

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    valid_size = int(.2 * X_train.shape[0])

    X_trainb = X_train[:-valid_size]
    Y_trainb = Y_train[:-valid_size]
    X_valid = X_train[-valid_size:]
    Y_valid = Y_train[-valid_size:]

    X_trainb, Y_trainb = preprocess_data(X_trainb, Y_trainb)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)
    X_test, Y_test = preprocess_data(X_test, Y_test)

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
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Lambda(lambda img: resize(img, (224, 224)))(inputs)
    x = base_model(x, training=False)
    gap = GlobalAveragePooling2D()(x)
    dropout1 = Dropout(0.5)(gap)
    fc1 = Dense(512, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(fc1)
    outputs = Dense(10, activation='softmax')(dropout2)
    model = models.Model(inputs, outputs)

    #compile
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'] 
        )


    #fit model
    history = model.fit(
        X_trainb, Y_trainb,
        validation_data = (X_valid, Y_valid),
        epochs=5,
        batch_size=64

    )
    #unfreeze after initial fitting/training
    for layer in base_model.layers[:-10]:
        layer.trainable = True

    #recompile
    model.compile(
        optimizer=Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    #continue training w early stopping
    earlystop = EarlyStopping(monitor="val_loss",
                              patience=4,
                              restore_best_weights=True)
    history_2 = model.fit(
        X_trainb, Y_trainb,
        validation_data= (X_test, Y_test),
        epochs=10,
        batch_size=32,
        callbacks=[earlystop]
        )
    model.save('cifar10.hf')
    print("Model cifar10.h5 saved")
