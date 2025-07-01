#!/usr/bin/env python3
""" module that creates an autoencoder """
import tensorflow.keras as K

def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ creates a sparse autoencoder

input_dims: integer containing the dimensions of the model input
hidden_layers: list containin the num of nodes for ea. hid layer in the
encoder, respectively the hidden layers should be reversed for the decoder
latent_dims: int containing the dimens of the latent space representation
lambtha: regularization parameter used for L1 regularization
on the encoded output

Returns: encoder, decoder, auto

encoder is the encoder model
decoder is the decoder model
auto is the full autoencoder model
The autoencoder model should be compiled using adam optimization and binary
cross-entropy loss
All layers should use a relu activation except for the last layer in the
decoder, which should use sigmoid"""
    # Create the encoder model
    reg = K.regularizers.l1(1e-5)
    input_layer = K.Input(shape=(input_dims,))
    x = input_layer
    for nodes in hidden_layers:
        x = K.layers.Dense(nodes, activation='relu',)(x)
    latent_layer = K.layers.Dense(latent_dims,
                                  activation='relu',
                                  activity_regularizer=reg)(x)
    encoder = K.Model(inputs=input_layer, outputs=latent_layer,
                      name='encoder')

    # create the decoder model

    latent_input = K.Input(shape=(latent_dims,))
    x = latent_input
    for nodes in reversed(hidden_layers):
        x = K.layers.Dense(nodes, activation='relu')(x)
    output_layer = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(inputs=latent_input, outputs=output_layer,
                      name='decoder')

    # create the full autoencoder model

    auto_input = K.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = K.Model(inputs=auto_input, outputs=decoded, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    # The encoder, decoder, and autoencoder models are returned
    return encoder, decoder, auto