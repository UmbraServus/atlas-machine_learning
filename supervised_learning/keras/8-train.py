#!/usr/bin/env python3
""" one hot module using keras """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True,
                shuffle=False):
    """ that trains a model using mini-batch gradient descent
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

        learning_rate_decay: bool tht indic8s if learnin r8 decay shld b used
            learning rate decay shld only b performed if val_data exists
            the decay should be performed using inverse time decay
            the learning r8 shld decay in a stepwise fashion after ea epoch
            each time the learning rate updates, Keras shld print a message

        alpha: initial learning rate

        decay_rate: decay rate

        save_best: bool indicating whether to save the model after ea epoch
        if it is the best
            model is considered the best if its validation loss is
            the lowest that the model has obtained

        filepath: file path where the model should be saved

        verbose: boolean determines if output shld be printed during training

        shuffle: boolean that determines whether to shuffle the batches
        every epoch. Normally, it is a good idea to shuffle, but for
        reproducibility, we have chosen to set the default to False.

    Returns: the History object generated after training the model """
    callbacks = []
    model = network
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(
            patience=patience
        ))

    if learning_rate_decay and validation_data:
        def lr_schedule(epoch):
            """learning rate schedule formula.
            args:
                epoch: number of passes thru data"""
            lr = alpha / (1 + decay_rate * epoch)
            return lr
        callbacks.append(
            K.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
            )
    if filepath:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_best_only=save_best
        ))

    return model.fit(
        data,
        labels,
        batch_size=batch_size,
        callbacks=callbacks,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        verbose=verbose
    )
