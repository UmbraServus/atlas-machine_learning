#!/usr/bin/env python3
""" Creates masks for transformer """
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation in a transformer.

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in), input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out), target sentence

    Returns:
        encoder_mask: tf.Tensor of shape (batch_size, 1, 1, seq_len_in),
        padding mask for encoder
        combined_mask: tf.Tensor of shape
        (batch_size, 1, seq_len_out, seq_len_out),
        look-ahead + target padding mask for decoder
        decoder_mask: tf.Tensor of shape (batch_size, 1, 1, seq_len_in),
        padding mask for 2nd attention block in decoder
    """

    # Encoder padding mask
    # 1 where padding, 0 where actual tokens
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    # expand dims to match attention shape (batch_size, 1, 1, seq_len_in)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask for 2nd attention block
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Look-ahead mask for target
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
        )
    # Shape (1, seq_len_out, seq_len_out)
    look_ahead_mask = look_ahead_mask[tf.newaxis, :, :]

    # Target padding mask
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Combine look-ahead mask and target padding mask
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask[:, :, :, :])

    return encoder_mask, combined_mask, decoder_mask
