#!/usr/bin/env python3
"""creates a TF-IDF embedding matrix"""
bag_of_words = __import__('0-bag_of_words').bag_of_words
import numpy as np


def tf_idf(sentences, vocab=None):
    """that creates a TF-IDF embedding:
    
    sentences: list of sentences to analyze
    vocab: list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used
    
    Returns: embeddings, features

    embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        s is the number of sentences in sentences
        f is the number of features analyzed
    features: list of the features used for embeddings"""

    def compute_idf(sentences, features):
        """Helper function to compute IDF values for features"""
        import math
        N = len(sentences)
        idf_values = np.zeros(len(features))
        for j, feature in enumerate(features):
            df = sum(1 for sentence in sentences if feature in sentence.lower().split())
            idf_values[j] = math.log((N + 1) / (df + 1))
        return idf_values

    # Get term frequency matrix and features
    tf_matrix, features = bag_of_words(sentences, vocab)
    s, f = tf_matrix.shape

    # Compute IDF values
    idf_values = compute_idf(sentences, features)

    # Compute TF-IDF matrix
    tf_idf_matrix = tf_matrix * idf_values

    # Row-normalize
    row_norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0  # avoid division by zero
    tf_idf_matrix = tf_idf_matrix / row_norms

    return tf_idf_matrix, features