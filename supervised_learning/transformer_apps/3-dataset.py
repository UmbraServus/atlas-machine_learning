#!/usr/bin/env python3
""" Dataset for Machine Translation """

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Dataset for Machine Translation with preprocessing for training."""

    def __init__(self, batch_size, max_len):
        """
        Initializes the Dataset instance.

        Args:
            batch_size: int, batch size for training/validation
            max_len: int, maximum number of tokens allowed per sentence
        """

        self.batch_size = batch_size
        self.max_len = max_len
        self.vocab_size = 2 ** 13  # 8192

        # Load raw datasets
        raw_train, raw_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True,
        )

        # Train tokenizers from the raw training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(raw_train)

        # Map tf_encode to tokenize datasets
        tokenized_train = raw_train.map(self.tf_encode)
        tokenized_valid = raw_valid.map(self.tf_encode)

        # Filter examples longer than max_len
        tokenized_train = tokenized_train.filter(
            lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                          tf.size(en) <= max_len)
        )
        tokenized_valid = tokenized_valid.filter(
            lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                          tf.size(en) <= max_len)
        )

        # Training dataset: cache, shuffle, padded batch, prefetch
        self.data_train = (
            tokenized_train
            .cache()
            .shuffle(buffer_size=20000)
            .padded_batch(batch_size, padded_shapes=([None], [None]))
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # Validation dataset: padded batch only
        self.data_valid = tokenized_valid.padded_batch(
            batch_size, padded_shapes=([None], [None])
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset of (pt, en) sentence pairs

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """

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
            pt_iterator(), vocab_size=self.vocab_size
        )

        tokenizer_en = tokenizer_en_base.train_new_from_iterator(
            en_iterator(), vocab_size=self.vocab_size
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a Portuguese sentence and its corresponding English sentence
        into token IDs suitable for the model.

        The tokenized sentences include start and end of sentence tokens:
        - Start token index: vocab_size
        - End token index: vocab_size + 1

        Args:
            pt: tf.Tensor containing Portuguese sentence
            en: tf.Tensor containing English sentence

        Returns:
            pt_tokens: np.ndarray containing Portuguese tokens
            en_tokens: np.ndarray containing English tokens
        """
        # convert bytes to string
        pt = pt.numpy().decode("utf-8")
        en = en.numpy().decode("utf-8")

        # Encode sentences using the tokenizers
        pt_ids = self.tokenizer_pt.encode(pt)
        en_ids = self.tokenizer_en.encode(en)

        # Start and end token IDs
        start_token = self.vocab_size
        end_token = self.vocab_size + 1

        pt_tokens = [start_token] + pt_ids + [end_token]
        en_tokens = [start_token] + en_ids + [end_token]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for encode method.
        Converts tf.Tensor input to token ID tensors
        suitable for tf.data.Dataset.

        Args:
            pt: tf.Tensor containing Portuguese sentence
            en: tf.Tensor containing English sentence

        Returns:
            pt_tokens: tf.Tensor of Portuguese token IDs
            en_tokens: tf.Tensor of English token IDs
        """
        # Call Python encode via tf.py_function
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set shape for TF graph compatibility
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
