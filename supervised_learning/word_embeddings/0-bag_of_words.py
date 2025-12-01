#!/usr/bin/env python3
"""creates a bag of words embedding matrix"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix

sentences: list of sentences to analyze
vocab: list of the vocabulary words to use for the analysis
If None, all words within sentences should be used

Returns: embeddings, features

embeddings: numpy.ndarray of shape (s, f) containing the embeddings
s is the number of sentences in sentences
f is the number of features analyzed

features: list of the features used for embeddings"""

    def normalize_text(text):
        """Helper function to normalize text"""
        if text.endswith('s') and len(text) > 3:
            return text[:-1]
        else:
            return text

    if vocab is None:
        # Create vocabulary from sentences
        vocab = set()
        for sentence in sentences:
            # Remove punctuation and special characters
            sentence = re.sub(r"[!?.,;:']", "", sentence)
            # lower case and split
            words = sentence.lower().split()
            # normalize
            words = [normalize_text(word) for word in words]
            vocab.update(words)
        vocab = sorted(vocab)  # Sort to ensure consistent order

    features = vocab
    f = len(features)
    s = len(sentences)
    embeddings = np.zeros((s, f), dtype=int)

    feature_index = {word: idx for idx, word in enumerate(features)}
    features = np.array(features)

    for i, sentence in enumerate(sentences):
        sentence = re.sub(r"[!?.,;:']", "", sentence)
        words = sentence.lower().split()
        words = [normalize_text(word) for word in words]
        for word in words:
            if word in feature_index:
                embeddings[i, feature_index[word]] += 1

    return embeddings, features
