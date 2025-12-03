#!/usr/bin/env python3
bag_of_words = __import__('0-bag_of_words').bag_of_words
"""creates a TF-IDF embedding matrix"""
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

    tf_matrix, features = bag_of_words(sentences, vocab)

    N = len(sentences)
    df = np.count_nonzero(tf_matrix > 0, axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1  # Smoothing

    tfidf = tf_matrix * idf

    # Row-normalize
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf_normalized = tfidf / norms

    return tfidf_normalized, features
