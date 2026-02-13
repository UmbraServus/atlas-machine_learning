#!/usr/bin/env python3
"""Train LeNet-5 on CIFAR-10"""
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from lenet5 import lenet5
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def preprocess(X):
    """normalize, and convert to grayscale"""
    X = tf.cast(X, tf.float32) / 255.0
    # X = tf.image.rgb_to_grayscale(X)
    return X


if __name__ == "__main__":
    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocess
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build model
    inputs = tf.keras.Input(shape=(32, 32, 3))
    model = lenet5(inputs)

    early_stop = EarlyStopping(
        monitor='val_loss',  # could also use 'val_accuracy'
        patience=10,  # how many epochs to wait for improvement
        restore_best_weights=True  # revert to the best weights when stopping
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    # Train
    model.fit(
        X_train,
        y_train,
        batch_size=128,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr]
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
