#!/usr/bin/env python3
""" one hot module using keras """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    """ that trains a model using mini-batch gradient descent:
    args:
        network: model to train
        data: np.ndarray shape (m, nx) containing the input data
        labels: one-hot np.ndarray shape (m, classes) containing labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        verbose: boolean determines if output shld be printed during training
        shuffle: boolean that determines whether to shuffle the batches
        every epoch. Normally, it is a good idea to shuffle, but for reproducibility,
        we have chosen to set the default to False.
    Returns: the History object generated after training the model """

    model = network
    model.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle,
        verbose=verbose
    )
    return model