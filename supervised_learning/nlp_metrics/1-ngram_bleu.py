#!/usr/bin/env python3
"""calculates the n-gram BLEU score for a sentence"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations
            each ref trnslation is a list of the words in the tnslation
        sentence: list containing the model proposed sentence
        n: size of the n-grams to use

    Returns:
        the n-gram BLEU score
    """

    # Helper to extract n-grams from a token list
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    # Candidate sentence length
    c = len(sentence)

    # Find closest reference length
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = ref_lengths[0]
    min_distance = abs(ref_lengths[0] - c)

    for ref_len in ref_lengths:
        distance = abs(ref_len - c)
        if distance < min_distance:
            min_distance = distance
            closest_ref_len = ref_len

    # Calculate Brevity Penalty (BP)
    if c > closest_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - closest_ref_len / c) if c > 0 else 0

    # Extract candidate n-grams and count them
    cand_ngrams = get_ngrams(sentence, n)
    total_cand_ngrams = len(cand_ngrams)

    candidate_counts = {}
    for ng in cand_ngrams:
        candidate_counts[ng] = candidate_counts.get(ng, 0) + 1

    # Extract reference n-gram maximum counts across references
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        ref_count = {}
        for ng in ref_ngrams:
            ref_count[ng] = ref_count.get(ng, 0) + 1
        # take max count over references
        for ng, cnt in ref_count.items():
            max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), cnt)

    # Compute clipped matching n-grams
    matching_ngrams = 0
    for ng, count in candidate_counts.items():
        matching_ngrams += min(count, max_ref_counts.get(ng, 0))

    # Precision
    precision = (matching_ngrams /
                 total_cand_ngrams if total_cand_ngrams > 0 else 0
    )

    # Final BLEU score
    bleu_score = BP * precision

    return bleu_score
