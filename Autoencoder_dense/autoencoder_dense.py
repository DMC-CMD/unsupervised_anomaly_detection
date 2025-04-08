from keras.src.losses import MeanSquaredError
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam
import os
import pickle


class AutoencoderDense:
    def __init__(self, input_shape, layer_output_sizes, bottleneck_size):
        self._input_shape = input_shape
        self._layer_output_sizes = layer_output_sizes
        self._bottleneck_size = bottleneck_size

        self._layer_amount = len(self._layer_output_sizes)
        self._flattened_input_size = input_shape[0] * input_shape[1]
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
        flatten = Flatten()
        layers = flatten(encoder_input)
        for layer_index in range(self._layer_amount):
            layer_number = layer_index + 1
            dense_layer = Dense(self._layer_output_sizes[layer_index], name='encoder_fully_connected_layer_' + str(layer_number))
            layers = dense_layer(layers)
            relu_layer = ReLU(name='encoder_relu_layer_' + str(layer_number))
            layers = relu_layer(layers)

        bottleneck = Dense(self._bottleneck_size, name='bottleneck')(layers)

        self._encoder = Model(encoder_input, bottleneck, name='encoder')

    def _build_decoder(self):
        decoder_input = Input(shape=(self._bottleneck_size, ), name='decoder_input')
        layers = decoder_input
        for layer_index in reversed(range(0, self._layer_amount)):
            layer_number = self._layer_amount - layer_index
            dense_layer = Dense(self._layer_output_sizes[layer_index], name='decoder_fully_connected_layer_' + str(layer_number))
            layers = dense_layer(layers)
            relu_layer = ReLU(name='decoder_relu_layer_' + str(layer_number))
            layers = relu_layer(layers)

        dense_layer = Dense(self._flattened_input_size)
        layers = dense_layer(layers)

        sigmoid = Activation("sigmoid", name="sigmoid_layer")
        layers = sigmoid(layers)

        output_layer = Reshape(self._input_shape)(layers)
        self._decoder = Model(decoder_input, output_layer, name='decoder')


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
        self._batch_size = batch_size
        self._epochs = epochs

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
            self._layer_output_sizes,
            self._bottleneck_size
        ]
        file_path = os.path.join(save_folder, ".parameters.pkl")
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, "wb") as file:
            pickle.dump(parameters, file)

    def get_hyperparameters(self):
        return self._layer_output_sizes, self._bottleneck_size

    def _save_train_parameters(self, save_folder):
        train_parameters = [
            self._batch_size,
            self._epochs,
        ]
        file_path = os.path.join(save_folder, ".train_parameters.pkl")
        if os.path.exists(file_path):
            os.remove(file_path)
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
        hyperparameters_file_path = os.path.join(save_folder, ".parameters.pkl")
        with open(hyperparameters_file_path, "rb") as file:
            hyperparameters = pickle.load(file)
        autoencoder = AutoencoderDense(*hyperparameters)

        train_parameters_file_path = os.path.join(save_folder, ".train_parameters.pkl")
        with open(train_parameters_file_path, "rb") as file:
            train_parameters = pickle.load(file)
        autoencoder._set_train_parameters(*train_parameters)

        weights_file_path = os.path.join(save_folder, ".weights.h5")
        autoencoder._load_weights(weights_file_path)
        return autoencoder

    def _load_weights(self, weights_file_path):
        self._model.load_weights(weights_file_path)

    def predict(self, data, verbose=0):
        verbose= 1 if verbose else 0
        return self._model.predict(data, verbose=verbose)
