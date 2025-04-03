from keras.src.losses import MeanSquaredError
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.optimizers import Adam
import os
import pickle


class AutoencoderConvolutional:
    def __init__(self, input_shape, filter_sizes, filter_amount, strides, bottleneck_size):
        self._input_shape = input_shape
        self._filter_sizes = filter_sizes
        self._filter_amounts = filter_amount
        self._strides = strides
        self._bottleneck_size = bottleneck_size

        self._layer_amount = len(self._filter_amounts)
        self._shape_before_bottleneck = None
        self._train_history = None
        self._batch_size = None
        self._epochs = None

        self._encoder = None
        self._decoder = None
        self._model = None

        self._build()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = Input(shape=self._input_shape, name='encoder_input')
        layers = encoder_input
        for layer_index in range(self._layer_amount):
            layers = self._append_convolutional_layer(layer_index, layers)

        self._shape_before_bottleneck = keras_backend.int_shape(layers)[1:]
        bottleneck = Flatten()(layers)
        bottleneck = Dense(self._bottleneck_size, name='encoder_output')(bottleneck)
        self._encoder = Model(encoder_input, bottleneck, name='encoder')

    def _append_convolutional_layer(self, layer_index, layers):
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self._filter_amounts[layer_index],
            kernel_size=self._filter_sizes[layer_index],
            strides=self._strides[layer_index],
            padding='same',
            name='encoder_conv_layer_' + str(layer_number)
        )
        layers = conv_layer(layers)
        layers = ReLU(name='encoder_relu_'+str(layer_number))(layers)
        layers = BatchNormalization(name='encoder_bn_' + str(layer_number))(layers)
        return layers

    def _build_decoder(self):
        decoder_input = Input(shape=(self._bottleneck_size,), name='decoder_input')
        neuron_amount = np.prod(self._shape_before_bottleneck)
        layers = Dense(neuron_amount, name='decoder_dense')(decoder_input)
        layers = Reshape(target_shape=self._shape_before_bottleneck)(layers)

        for layer_index in reversed(range(1, self._layer_amount)):
            layers = self._append_conv_transpose_layer(layer_index, layers)

        convolutional_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self._filter_sizes[0],
            strides=self._strides[0],
            padding='same'
        )
        layers = convolutional_transpose_layer(layers)

        output_layer = Activation("sigmoid", name="sigmoid_layer")
        layers = output_layer(layers)
        self._decoder = Model(decoder_input, layers, name='decoder')

    def _append_conv_transpose_layer(self, layer_index, layers):
        layer_number = self._layer_amount - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self._filter_amounts[layer_index],
            kernel_size=self._filter_sizes[layer_index],
            strides=self._strides[layer_index],
            padding='same',
            name='decoder_conv_transpose_layer_'+str(layer_number)
        )
        layers = conv_transpose_layer(layers)
        layers = ReLU(name='decoder_relu_'+str(layer_number))(layers)
        layers = BatchNormalization(name='decoder_bn_'+str(layer_number))(layers)
        return layers

    def _build_autoencoder(self):
        input = Input(shape=self._input_shape, name='autoencoder_input')
        layers = self._decoder(self._encoder(input))
        self._model = Model(input, layers, name='autoencoder')

    def summary(self):
        self._encoder.summary()
        self._decoder.summary()
        self._model.summary()

    def compile(self, learning_rate=0.0001, loss_function=MeanSquaredError()):
        optimizer = Adam(learning_rate=learning_rate)
        self._model.compile(optimizer=optimizer, loss=loss_function)

    def train(self, train_data, validation_data, batch_size, epochs, verbose):
        verbose = 1 if verbose else 0
        history = self._model.fit(
            train_data, train_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(validation_data, validation_data),
            shuffle=True,
            verbose=verbose
        )
        self._train_history = history
        self._batch_size = batch_size
        self._epochs = epochs

        return history

    def get_history(self):
        return self._train_history

    def save(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self._save_hyperparameters(save_folder)
        self._save_train_parameters(save_folder)
        self._save_weights(save_folder)

    def _save_hyperparameters(self, save_folder):
        parameters = [
            self._input_shape,
            self._filter_sizes,
            self._filter_amounts,
            self._strides,
            self._bottleneck_size
        ]
        file_path = os.path.join(save_folder, ".parameters.pkl")
        if os.path.exists(file_path):
            os.remove(file_path)

        with open(file_path, "wb") as file:
            pickle.dump(parameters, file)

    def get_hyperparameters(self):
        return self._filter_amounts, self._bottleneck_size

    def _save_train_parameters(self, save_folder):
        train_parameters = [
            self._batch_size,
            self._epochs
        ]
        file_path = os.path.join(save_folder, ".train_parameters.pkl")
        with open(file_path, "wb") as file:
            pickle.dump(train_parameters, file)

    def _set_train_parameters(self, batch_size, epochs):
        self._batch_size = batch_size
        self._epochs = epochs

    def get_train_parameters(self):
        return self._batch_size, self._epochs

    def _save_weights(self, save_folder):
        file_path = os.path.join(save_folder, ".weights.h5")
        if os.path.exists(file_path):
            os.remove(file_path)
        self._model.save_weights(file_path)

    @classmethod
    def load(cls, save_folder):
        parameters_file_path = os.path.join(save_folder, ".parameters.pkl")
        with open(parameters_file_path, "rb") as file:
            parameters = pickle.load(file)
        autoencoder = AutoencoderConvolutional(*parameters)

        train_parameters_file_path = os.path.join(save_folder, ".train_parameters.pkl")
        with open(train_parameters_file_path, "rb") as file:
            train_parameters = pickle.load(file)
        autoencoder._set_train_parameters(train_parameters[0], train_parameters[1])

        weights_file_path = os.path.join(save_folder, ".weights.h5")
        autoencoder._load_weights(weights_file_path)
        return autoencoder

    def _load_weights(self, weights_file_path):
        self._model.load_weights(weights_file_path)

    def predict(self, data, verbose):
        verbose = 1 if verbose else 0
        return self._model.predict(data, verbose=verbose)
