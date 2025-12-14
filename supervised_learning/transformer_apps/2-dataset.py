#!/usr/bin/env python3
""" Dataset for Machine Translation """

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


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
        raw_train, raw_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True,
        )

        # Create tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            raw_train
        )

        # Map the TensorFlow wrapper to tokenize the datasets
        self.data_train = raw_train.map(self.tf_encode)
        self.data_valid = raw_valid.map(self.tf_encode)

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
        tokenizer_pt_base = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en_base = transformers.AutoTokenizer.from_pretrained(
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


    def encode(self, pt, en):
        """Encodes a Portuguese sentence and its corresponding English sentence
    into token IDs suitable for the model.

    The tokenized sentences include start and end of sentence tokens:
    - Start token index: vocab_size
    - End token index: vocab_size + 1

    Args:
        pt: string containing the Portuguese sentence
        en: string containing the English sentence

    Returns:
        pt_tokens: np.ndarray containing the Portuguese tokens
        en_tokens: np.ndarray containing the English tokens
    """
        # convert bytes to string
        pt = pt.numpy().decode("utf-8")
        en = en.numpy().decode("utf-8")
        # Encode the Portuguese sentence using the Portuguese tokenizer
        pt_ids = self.tokenizer_pt.encode(pt)
        # Encode the English sentence using the English tokenizer
        en_ids = self.tokenizer_en.encode(en)

        # start and end token IDs
        start_pt = self.tokenizer_pt.vocab_size
        end_pt = self.tokenizer_pt.vocab_size + 1

        pt_tokens = [start_pt] + pt_ids + [end_pt]
        en_tokens = [start_pt] + en_ids + [end_pt]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for encode method.
        Converts tf.Tensor input to token ID tensors suitable for tf.data.Dataset.

        Args:
            pt: tf.Tensor containing Portuguese sentence
            en: tf.Tensor containing English sentence

        Returns:
            pt_tokens: tf.Tensor of Portuguese token IDs
            en_tokens: tf.Tensor of English token IDs
        """

        # Use tf.py_function to call Python encode function
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set shape for TF graph compatibility
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
