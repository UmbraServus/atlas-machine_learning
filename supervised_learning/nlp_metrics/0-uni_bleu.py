#!/usr/bin/env python3
"""calculates the unigram BLEU score for a sentence"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    To compute precision, one simply counts up the number of candidate
    translation words (unigrams) which occur in any reference translation
    and then divides by the total number of words in the candidate trnslation

    Args:
        references: list of reference translations
        each reference translation is a list of the words in the translation
        sentence: list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    # Candidate length
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

    # Calculate Brevity Penalty
    if c > closest_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - closest_ref_len / c)

    # Count unigrams in the candidate sentence
    candidate_counts = {}
    for word in sentence:
        candidate_counts[word] = candidate_counts.get(word, 0) + 1

    # For each word in candidate, find max count across ALL references
    matching_unigrams = 0
    for word, count in candidate_counts.items():
        # Find maximum count of this word in any reference
        max_ref_count = 0
        for reference in references:
            ref_count = reference.count(word)
            max_ref_count = max(max_ref_count, ref_count)

        # Clip to the maximum found in references
        matching_unigrams += min(count, max_ref_count)

    # Calculate precision
    precision = matching_unigrams / c if c > 0 else 0

    # Calculate BLEU score (precision with brevity penalty)
    bleu_score = BP * precision

    return bleu_score
