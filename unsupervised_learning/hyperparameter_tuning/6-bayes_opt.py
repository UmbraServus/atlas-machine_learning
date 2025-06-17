#!/usr/bin/env python3
"""Module for Bayesian Optimization of a Keras model using GPyOpt"""
import numpy as np
import tensorflow as tf
import GPyOpt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os



# 2. Load and preprocess the dataset
#Load and preprocess the MNIST dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_val = x_val.reshape(-1, 28 * 28).astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

# Track the best validation accuracy and its model filename
best_val_acc = 0
best_model_path = ""

# 3. Define the function to create, train, and evaluate the model
def objective_function(params):
    """objective function for Bayesian Optimization
    params: list of hyperparameters to tune
    Each element in params is a list containing:
        [learning_rate, units, dropout_rate, l2_weight, batch_size]
    returns: validation accuracy after training the model"""
    global best_val_acc, best_model_path
    learning_rate = float(params[0][0])
    units = int(params[0][1])
    dropout_rate = float(params[0][2])
    l2_weight = float(params[0][3])
    batch_size = int(params[0][4])
    # Build the Keras model
    model = Sequential([
        Dense(units, activation='relu',
              input_shape=(28 * 28,),
              kernel_regularizer=l2(l2_weight)),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
#       - Setup callbacks: EarlyStopping and ModelCheckpoint (filename includes hyperparams)
    checkpt_name = f'lr{learning_rate}_units{units}_dropout{dropout_rate}_l2{l2_weight}_batch{batch_size}.h5'
    callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ModelCheckpoint(checkpt_name, monitor='val_accuracy', save_best_only=True)
    ]
#       - Train the model on training data, validate on validation data
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=32,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0)
#       - Return validation loss/error after training
    val_acc = max(history.history['val_accuracy'])
    print(
    f"Trained with: "
    f"learning_rate={learning_rate:.5f}, "
    f"units={units}, "
    f"dropout_rate={dropout_rate:.2f}, "
    f"l2_weight={l2_weight:.5f}, "
    f"batch_size={batch_size} "
    f"=> Validation Accuracy: {val_acc:.4f}"
    )
    # Save only best model
    if val_acc > best_val_acc:
        # Remove previous best (if it exists)
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)

        best_val_acc = val_acc
        best_model_path = checkpt_name
    else:
        # Delete this model, since it's not the best
        if os.path.exists(checkpt_name):
            os.remove(checkpt_name)

    return -val_acc
# 4. Define the domain (search space) for hyperparameters to tune
#    Example: learning rate between 0.0001 and 0.1,
#             units in hidden layer between 10 and 200,
#             dropout rate between 0 and 0.5,
#             L2 regularization weight between 0 and 0.1,
#             batch size from 16 to 128
domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-3, 0.1)},
    {'name': 'units', 'type': 'discrete', 'domain': (32, 64, 128)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.3)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-6, 0.01)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
]

# 5. Setup Bayesian Optimization with GPyOpt
#    - Objective function is the function defined in step 3
#    - Domain is from step 4
#    - Max iterations set to 30
bo = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=domain,
    acquisition_type='EI',  # Expected Improvement
    exact_eval=True,)
# 6. Run Bayesian Optimization
bo.run_optimization(max_iter=30)
# 7. After optimization finishes:
#    - Extract best hyperparameters and best value of satisficing metric
#    - Plot convergence graph of optimization (objective value vs iterations)
#    - Save a report (best hyperparams, best metric, optimization history) to 'bayes_opt.txt'
bo.plot_convergence()
val_accs = -bo.Y.flatten()  # bo.Y is an array of negative validation accuracies
plt.plot(np.arange(1, len(val_accs) + 1), val_accs, marker='o')
plt.title('Validation Accuracy over Bayesian Optimization Iterations')
plt.xlabel('Iteration')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()

with open('bayes_opt.txt', 'w') as f:
    f.write(str(bo.fx_opt) + '\n')
    f.write(str(bo.x_opt) + '\n')
# 8. End
print("Best hyperparams:")
print(f"  Learning rate:     {bo.x_opt[0]:.5f}")
print(f"  Units:             {int(bo.x_opt[1])}")
print(f"  Dropout rate:      {bo.x_opt[2]:.5f}")
print(f"  L2 weight:         {bo.x_opt[3]:.5f}")
print(f"Batch size:{int(bo.x_opt[4])}")
print(f"Best validation acc: {-bo.fx_opt:.4f}")
