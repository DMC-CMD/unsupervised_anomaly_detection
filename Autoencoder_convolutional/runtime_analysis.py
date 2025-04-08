import timeit
import numpy as np
from sklearn.metrics import mean_squared_error
from autoencoder_convolutional import AutoencoderConvolutional
from evaluation_helper import transform_to_spectrogram


def run_inference(input_file, autoencoder):
    normal_test = np.load(input_file)
    normal_test = normal_test[:150]

    reconstructed_normals = autoencoder.predict(normal_test)
    reconstructed_normals = transform_to_spectrogram(reconstructed_normals)
    normal_test = transform_to_spectrogram(normal_test)

    print(reconstructed_normals.shape)
    threshold = 0.0016 # for the runtime evaluation, the specific threshold is not relevant and is therefore fixed
    classifications = []
    for i in range(len(reconstructed_normals)):
        current_error = mean_squared_error(normal_test[i], reconstructed_normals[i])
        if current_error >= threshold:
            classifications.append(1)
        else:
            classifications.append(0)

if __name__ == "__main__":

    executions = 20

    # load the specific autoencoders you want to use for the runtime analysis
    tuned_model_hydropower = AutoencoderConvolutional.load('Models/Hydropower/10')
    t = timeit.Timer(lambda : run_inference('../Spectrograms/Hydropower/normal_test_spectrograms.npy', tuned_model_hydropower))
    r = t.repeat(executions, 1)
    print('Best runtime (s) to classify 150 Hydropower spectrograms with Convolutional Autoencoder: ', min(r))

    tuned_model_mimii = AutoencoderConvolutional.load('Models/MIMII/2')
    t = timeit.Timer(lambda: run_inference('../Spectrograms/MIMII/normal_test_spectrograms.npy', tuned_model_mimii))
    r = t.repeat(executions, 1)
    print('Best runtime (s) to classify 150 MIMII spectrograms with Convolutional Autoencoder: ', min(r))



