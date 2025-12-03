#!/usr/bin/env python3
""" calculates the ngram BLEU score for a sentence """
import numpy as np


def ngram_bleu(references, sentence, n):
    """that calculates the n-gram BLEU score for a sentence:

references: list of reference translations
each reference translation is a list of the words in the translation
sentence: list containing the model proposed sentence
n: the size of the n-gram to use for evaluation

Returns: the n-gram BLEU score"""
