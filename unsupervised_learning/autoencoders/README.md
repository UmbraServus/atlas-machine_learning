autoencoders

0-vanilla.py - Vanilla Autoencoder that simply encodes and decodes in one model as an autoencoder. returns encoder, decoder, autoencoder.

1-sparse.py - similar to autoencoder but with regularization to learn more unique features on the reconstruction by dropping out some of the nodes in the encoding.

2-convolutional - convolutional autoencoder; note that it doesnt use every hidden_layer in the decoder cause it doesnt need the last hidden_layer nodes in the filters list for reconstruciton.