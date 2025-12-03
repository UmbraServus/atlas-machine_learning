#!/usr/bin/env python3
"""creates a fasttext gensim model"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds and trains a gensim fasttext model

    Args:
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word for use in trainin
        window: maximum distance between current and predicted word
        negative: size of negative sampling
        cbow: True for CBOW, False for Skip-gram
        epochs: number of iterations to train over
        seed: seed for the random number generator
        workers: number of worker threads to train the model

    Returns:
        the trained model
    """
    # Determine skip_gram parameter: 0 for CBOW, 1 for Skip-gram
    skip_gram = 0 if cbow else 1

    # Create the Word2Vec model
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        sg=skip_gram,
        negative=negative,
        seed=seed,
        workers=workers
    )

    # Build vocabulary
    model.build_vocab(sentences)

    # Train the model
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
