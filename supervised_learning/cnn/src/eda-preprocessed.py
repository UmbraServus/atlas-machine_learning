#!/usr/bin/env python3
""" module for EDA, feature selection, feature engineering, and training w/
cifar-10"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape, y_train.shape)

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

X_train_resized = tf.image.resize(X_train, (28, 28))
X_test_resized = tf.image.resize(X_test, (28, 28))
X_train_normalized = X_train_resized / 255.0
X_test_normalized = X_test_resized / 255.0
plt.figure(figsize=(4, 4))

for i in range(9):
    idx = np.random.randint(0, len(X_train_normalized))
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train_normalized[idx], interpolation='nearest')
    plt.title(class_names[y_train[idx][0]])
    plt.axis('off')
plt.tight_layout()
plt.show()
X_train_np = X_train_normalized.numpy()
plt.hist(X_train_np.ravel(), bins=50)
plt.title("Pixel Value Distribution")
plt.show()
variances = X_train_np.var(axis=(1,2,3))

plt.hist(variances, bins=50)
plt.title("Pixel Variance Distribution")
plt.xlabel("Variance")
plt.ylabel("Image Count")
plt.show()

for i, channel in enumerate(['Red', 'Green', 'Blue']):
    plt.hist(X_train_np[:, :, :, i].ravel(), bins=50)
    plt.title(f'{channel} Channel Distribution')
    plt.show()