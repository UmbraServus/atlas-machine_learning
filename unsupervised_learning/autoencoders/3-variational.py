#!/usr/bin/env python3
""" module that creates a variational autoencoder """
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates a variational autoencoder

input_dims: int containing the dimensions of the model input
hidden_layers: list containing the # of nodes for each hid layer in the
encoder, respectively the hidden layers should be reversed for the decoder
latent_dims: int containing the dimensions of the latent space representation

Returns: encoder, decoder, auto

encoder is the encoder model, which should output the latent representation,
the mean, and the log variance, respectively

decoder is the decoder model

auto is the full autoencoder model

The autoencoder model should be compiled using adam optimization and
binary cross-entropy loss

All layers should use a relu activation except for the mean and
log variance layers in the encoder, which should use None,
and the last layer in the decoder, which should use sigmoid"""

    # Create the encoder model
    input_layer = K.Input(shape=(input_dims,))
    x = input_layer
    
    # Add hidden layers
    for nodes in hidden_layers:
        x = K.layers.Dense(nodes, activation='relu')(x)
    
    # Create mean and log variance layers (not from latent_layer)
    mu = K.layers.Dense(latent_dims, activation=None)(x)
    log_var = K.layers.Dense(latent_dims, activation=None)(x)
    
    # Sampling function
    def sampling(args):
        mu, log_var = args
        batch_size = K.backend.shape(mu)[0]
        epsilon = K.backend.random_normal(shape=(batch_size, latent_dims))
        return mu + K.backend.exp(0.5 * log_var) * epsilon
    
    # Sample from the latent space
    z = K.layers.Lambda(sampling, output_shape=(latent_dims,))([mu, log_var])
    
    encoder = K.Model(inputs=input_layer, outputs=[z, mu, log_var],
                      name='encoder')
    
    # Create the decoder model
    latent_input = K.Input(shape=(latent_dims,))
    x = latent_input
    
    # Add hidden layers in reverse order
    for nodes in reversed(hidden_layers):
        x = K.layers.Dense(nodes, activation='relu')(x)
    
    # Output layer with sigmoid activation
    output_layer = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(inputs=latent_input, outputs=output_layer,
                      name='decoder')
    
    # Create the full autoencoder model
    auto_input = K.Input(shape=(input_dims,))
    z, mu, log_var = encoder(auto_input)
    decoded = decoder(z)
    auto = K.Model(inputs=auto_input, outputs=decoded, name='autoencoder')

    # Add KL divergence loss
    kl_loss = -0.5 * K.backend.sum(1 + log_var - K.backend.square(mu) -
                                    K.backend.exp(log_var), axis=-1)
    kl_loss = K.backend.mean(kl_loss)
    auto.add_loss(kl_loss)
    
    # Compile the autoencoder model
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto