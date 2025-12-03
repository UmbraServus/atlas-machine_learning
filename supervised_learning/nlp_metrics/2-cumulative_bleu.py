#!/usr/bin/env python3
""" calculates the culminative BLEU score for a sentence """
import numpy as np


def cumulative_bleu(references, sentence, n):
    """that calculates the cumulative n-gram BLEU score for a sentence:

references: list of reference translations
each reference translation is a list of the words in the translation
sentence: list containing the model proposed sentence
n: the size of the largest n-gram to use for evaluation
All n-gram scores should be weighted evenly

Returns: the cumulative n-gram BLEU score"""
