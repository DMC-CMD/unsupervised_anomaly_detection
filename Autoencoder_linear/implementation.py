import numpy as np

from autoencoder_linear import AutoencoderLinear
from evaluation_helper import display_training_process


def train_linear_ae(first_layer_output_size, bottleneck_size, batch_size, epochs, dataset, verbose=False):
    input_dir = f'../Spectrograms/{dataset}'
    normal_train = np.load(input_dir + '/normal_train_spectrograms.npy')
    normal_validation = np.load(input_dir + '/normal_validation_spectrograms.npy')
    input_shape = normal_train[0].shape

    autoencoder = AutoencoderLinear(
        input_shape=input_shape,
        layer_output_sizes=(
            first_layer_output_size,
            int(first_layer_output_size/2),
            int(first_layer_output_size/4)),
        bottleneck_size=bottleneck_size
    )
    autoencoder.compile(0.0001)

    autoencoder.train(
        train_data=normal_train,
        validation_data=normal_validation,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose
    )

    return autoencoder


if __name__ == '__main__':

    linear_ae_hydropower = train_linear_ae(64, 8, 64, 10, dataset='Hydropower', verbose=True)
    linear_ae_hydropower.summary()
    linear_ae_hydropower.save('Models/Hydropower/base_model')
    display_training_process(linear_ae_hydropower.get_history(), 'Linear AE training with Hydropower data')


   
    linear_ae_mimii = train_linear_ae(64, 8, 64, 75, dataset='MIMII', verbose=True)
    linear_ae_mimii.summary()
    linear_ae_mimii.save('Models/MIMII/base_model')
    display_training_process(linear_ae_mimii.get_history(), 'Linear AE training with MIMII data')
    


