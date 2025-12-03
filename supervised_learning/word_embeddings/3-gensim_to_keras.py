#!/usr/bin/env python3
""" converts a gensim word2vec model to a keras embedding matrix"""
import numpy as np
import keras
from keras.layers import Embedding

def gensim_to_keras(model):
    """Get a Keras 'Embedding' layer with weights set from Word2Vec model's
    learned word embeddings.

    Returns:
    `keras.layers.Embedding`
        Embedding layer, to be used as input to deeper network layers.

    """
    word_vectors= model.wv  # structure holding the result of training
    weights = word_vectors.vectors  # vectors themselves, a 2D numpy array    
    words = word_vectors.index_to_key  # which row in `weights` corresponds to which word?

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )
    return layer