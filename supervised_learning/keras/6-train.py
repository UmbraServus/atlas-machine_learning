#!/usr/bin/env python3
""" one hot module using keras """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """ that trains a model using mini-batch gradient descent:
    args:
        network: model to train

        data: np.ndarray shape (m, nx) containing the input data

        labels: one-hot np.ndarray shape (m, classes) containing labels

        batch_size: size of the batch used for mini-batch gradient descent

        epochs: number of passes through data for mini-batch gradient descent

        validation_data: data to validate the model with, if not None

        early_stopping: bool tht indicates whether early stopping shld b used
            early stopping should only be performed if validation_data exists
            early stopping should be based on validation loss

        patience: patience used for early stopping

        verbose: boolean determines if output shld be printed during training

        shuffle: boolean that determines whether to shuffle the batches
        every epoch. Normally, it is a good idea to shuffle, but for
        reproducibility, we have chosen to set the default to False.

    Returns: the History object generated after training the model """

    model = network
    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(
            patience=patience,
            verbose=1
        )
    return model.fit(
        data,
        labels,
        batch_size=batch_size,
        callbacks=[callback],
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        verbose=verbose
    )
