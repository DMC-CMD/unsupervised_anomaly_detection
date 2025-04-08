import numpy as np

from autoencoder_convolutional import AutoencoderConvolutional
from evaluation_helper import display_roc, get_labels_and_reconstruction_errors, \
    get_auc_score, display_error_plot

autoencoder = AutoencoderConvolutional.load('Models/MIMII/base_model')

normal_test = np.load('../Spectrograms/MIMII/normal_test_spectrograms.npy')
anomaly_test = np.load('../Spectrograms/MIMII/anomaly_test_spectrograms.npy')

reconstructed_normal = autoencoder.predict(normal_test, verbose=1)
reconstructed_anomalies = autoencoder.predict(anomaly_test, verbose=1)

labels, reconstruction_errors = get_labels_and_reconstruction_errors(normal_test, anomaly_test, reconstructed_normal, reconstructed_anomalies)

display_error_plot(labels, reconstruction_errors, 'Reconstruction errors convolutional AE with MIMII data', y_label='Reconstruction error')
display_roc(labels, reconstruction_errors, 'ROC convolutional AE with Hydropower data', 'convolutional Autoencoder')

auc_score = get_auc_score(labels, reconstruction_errors)
print('ROC-AUC: ', auc_score)

