#!/usr/bin/env python3
"""module for lenet5 using v1 tensorflow"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):
    """ builds a mod ver of the LeNet-5 architecture using tensorflow
args:
    x: tf.placeholder shape (m, 28, 28, 1) containing input images
        m is the number of images
    y: tf.placeholder shape (m, 10) containing the one-hot labels

The model should consist of the following layers in order:
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes

All layers requiring initialization should initialize their kernels with
the he_normal initialization
method: tf.keras.initializers.VarianceScaling(scale=2.0)

All hidden layers requiring activation should use the relu activation
function

Returns:
tensor for the softmax activated output
training op that utilizes Adam opt (with default hyperparameters)
tensor for the loss of the netowrk
tensor for the accuracy of the network"""
    Relu = tf.nn.relu
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.layers.conv2d(x,
                             filters=6,
                             kernel_size=5,
                             padding='same',
                             activation=Relu,
                             kernel_initializer=initializer)

    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    conv2 = tf.layers.conv2d(pool1,
                             filters=16,
                             kernel_size=5,
                             padding='valid',
                             activation=Relu,
                             kernel_initializer=initializer)

    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
    flatten = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(flatten,
                          units=120,
                          kernel_initializer=initializer,
                          activation=Relu)
    fc2 = tf.layers.dense(fc1,
                          units=84,
                          kernel_initializer=initializer,
                          activation=Relu)
    logits = tf.layers.dense(fc2,
                             units=10,
                             kernel_initializer=initializer)

    output = tf.nn.softmax(logits)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        )
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return output, train_op, loss, acc
