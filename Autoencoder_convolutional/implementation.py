from evaluation_helper import display_training_process
from autoencoder_convolutional import AutoencoderConvolutional
import numpy as np


def train_conv_ae(first_layer_filter_amount, bottleneck_size, batch_size, epochs, dataset, verbose=False):
    input_folder = f'../Spectrograms/{dataset}'
    normal_train_spectrograms = np.load(input_folder + '/normal_train_spectrograms.npy')
    normal_validation_spectrograms = np.load(input_folder + '/normal_validation_spectrograms.npy')

    autoencoder = AutoencoderConvolutional(
        input_shape=normal_train_spectrograms[0].shape,
        filter_amounts=(
            first_layer_filter_amount,
            first_layer_filter_amount*2,
            first_layer_filter_amount*4
        ),
        filter_sizes=(3, 3, 3),
        strides=(2, 2, 2),
        bottleneck_size=bottleneck_size
    )

    autoencoder.compile(0.0001)

    autoencoder.train(
        normal_train_spectrograms,
        normal_validation_spectrograms,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    return autoencoder



if __name__ == '__main__':


    conv_autoencoder_hydropower = train_conv_ae(16, 64, 64, 25, 'Hydropower', verbose=True)
    conv_autoencoder_hydropower.save('Models/Hydropower/base_model')
    conv_autoencoder_hydropower.summary()
    display_training_process(conv_autoencoder_hydropower.get_history(), title='Convolutional AE Training with Hydropower data')


    conv_autoencoder_mimii = train_conv_ae( 16, 64, 64, 75, 'Mimii', verbose=True)
    conv_autoencoder_mimii.save('Models/MIMII/base_model')
    conv_autoencoder_mimii.summary()
    display_training_process(conv_autoencoder_mimii.get_history(), title='Convolutional AE Training with MIMII data')











