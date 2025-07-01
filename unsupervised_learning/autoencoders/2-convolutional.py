#!/usr/bin/env python3
""" module that creates a convolutional autoencoder """
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """ creates a convolutional autoencoder:

input_dims: tuple of integers containing the dimensions of the model input

filters: list containing the number of filters for each convolutional layer
in the encoder, respectively the filters should be reversed for the decoder

latent_dims: tuple of integers containing the dimensions of the latent space
representation

Each convolution in the encoder should use a kernel size of (3, 3) with same
padding and relu activation, followed by max pooling of size (2, 2)

Each convolution in the decoder, except for the last two, should use a filter
size of (3, 3) with same padding and relu activation, followed by upsampling of size (2, 2)

The second to last convolution should instead use valid padding

The last convolution should have the same number of filters as the number of
channels in input_dims with sigmoid activation and no upsampling

Returns: encoder, decoder, auto

encoder is the encoder model
decoder is the decoder model
auto is the full autoencoder model
The autoencoder model should be compiled using adam optimization and
binary cross-entropy loss"""

    #encoder model
    input_layer = K.Input(shape=input_dims)
    x = input_layer
    for f in filters:
        x = K.layers.Conv2D(filters=f, kernel_size=(3, 3),
                            padding='same',
                            activation='relu')(x)
        x = K.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    latent_layer = x
    #K.layers.Conv2D(filters=latent_dims[-1],
                                   #kernel_size=(3, 3),
                                   #padding='same',
                                   #activation='relu')(x)
    encoder = K.Model(inputs=input_layer, outputs=latent_layer, name='encoder')

    #decoder model
    latent_input = K.Input(shape=latent_dims)
    x = latent_input
    for f in reversed(filters[-1:]):
        x = K.layers.Conv2D(filters=f, kernel_size=(3, 3),
                            padding='same',
                            activation='relu')(x)
        x = K.layers.UpSampling2D(size=(2, 2))(x)
    x = K.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                        padding='valid',
                        activation='relu')(x)
    x = K.layers.UpSampling2D(size=(2, 2))(x)
    output_layer = K.layers.Conv2D(filters=input_dims[-1],
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='sigmoid')(x)
    decoder = K.Model(inputs=latent_input, outputs=output_layer, name='decoder')

    #autoencoder model
    auto_input = K.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = K.Model(inputs=auto_input, outputs=decoded, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    # The encoder, decoder, and autoencoder models are returned
    return encoder, decoder, auto