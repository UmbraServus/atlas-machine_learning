#!/usr/bin/env python3
""" Dataset for Machine Translation """

import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """Dataset for Machine Translation"""

    def __init__(self):
        """
        Creates the instance attributes:
        - data_train: ted_hrlr_translate/pt_to_en train split
        - data_valid: ted_hrlr_translate/pt_to_en validation split
        - tokenizer_pt: Portuguese tokenizer trained from dataset
        - tokenizer_en: English tokenizer trained from dataset
        """

        # Load datasets
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True,
        )

        # Create tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset

        Args:
            data: tf.data.Dataset of (pt, en) sentence pairs

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """

        vocab_size = 2 ** 13  # 8192

        # Load pretrained tokenizers
        tokenizer_pt_base = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en_base = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        # Sentence generators
        def pt_iterator():
            for pt, _ in data:
                yield pt.numpy().decode("utf-8")

        def en_iterator():
            for _, en in data:
                yield en.numpy().decode("utf-8")

        # Train new tokenizers from dataset
        tokenizer_pt = tokenizer_pt_base.train_new_from_iterator(
            pt_iterator(), vocab_size=vocab_size
        )

        tokenizer_en = tokenizer_en_base.train_new_from_iterator(
            en_iterator(), vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en
