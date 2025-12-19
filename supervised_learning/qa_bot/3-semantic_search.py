#!/usr/bin/env python3
""" semantic search for reference docs """
import tensorflow_hub as hub
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of docs.

    Args:
        corpus_path: Path to the corpus of reference docs
        sentence: The sentence from which to perform semantic search

    Returns:
        The reference text of the document most similar to sentence
    """
    # Load Universal Sentence Encoder from TensorFlow Hub
    # This model converts sentences into embeddings (numerical vectors)
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # Read all docs from the corpus
    docs = []

    # Check if corpus_path is a directory or a file
    if os.path.isdir(corpus_path):
        # Read all text files in the directory
        for filename in os.listdir(corpus_path):
            filepath = os.path.join(corpus_path, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        docs.append(f.read())
                except:
                    # Skip files that can't be read
                    continue
    else:
        # Assume it's a single file w/ docs separated by \n or some delim
        with open(corpus_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by double newlines (common document separator)
            docs = [doc.strip() for doc in content.split('\n\n') if doc.strip()]

    if not docs:
        return None

    # Get embedding for the query sentence
    query_embedding = embed([sentence])[0]

    # Get embeddings for all docs in the corpus
    corpus_embeddings = embed(docs)

    # Calculate cosine similarity between query and all docs
    # Cosine similarity = dot product of normalized vectors
    similarities = np.inner(query_embedding, corpus_embeddings)

    # Find the index of the most similar document
    most_similar_idx = np.argmax(similarities)

    # Return the most similar document
    return docs[most_similar_idx]
