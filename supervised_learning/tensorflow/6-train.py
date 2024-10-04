#!/usr/bin/env python3
"""module for learning tensorflow and setting up a simple neural network."""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
    X_train: numpy.ndarray containing the training input data
    Y_train: numpy.ndarray containing the training labels
    X_valid: numpy.ndarray containing the validation input data
    Y_valid: numpy.ndarray containing the validation labels
    layer_sizes: list containing the number of nodes in 
    each layer of the network
    activations: list containing the activation functions 
    for each layer of the network
    alpha: learning rate
    iterations: number of iterations to train over
    save_path: path to save the model

    Returns:
    The path where the model was saved.
    """
    graph = tf.Graph()
    with graph.as_default():
        x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
        y_pred = forward_prop(x, layer_sizes, activations)
        loss = calculate_loss(y, y_pred)
        accuracy = calculate_accuracy(y, y_pred)
        train_op = create_train_op(loss, alpha)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
    
        for i in range(iterations + 1):
            if i > 0:
                _, cost = sess.run(
                    [train_op, loss],
                    feed_dict={x: X_train, y: Y_train}
                    )
            else:
                training_cost = sess.run(
                    loss,
                    feed_dict={x: X_train, y: Y_train}
                    )
        
            training_accuracy = sess.run(
                accuracy,
                feed_dict={x: X_train, y: Y_train}
                )
            validation_cost = sess.run(
                loss,
                feed_dict={x: X_valid, y: Y_valid}
                )
            validation_accuracy = sess.run(
                accuracy,
                feed_dict={x: X_valid, y: Y_valid}
                )

            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {training_cost}")
                print(f"\tTraining Accuracy: {training_accuracy}")
                print(f"\tValidation Cost: {validation_cost}")
                print(f"\tValidation Accuracy: {validation_accuracy}")
    
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, save_path)
    return save_path
