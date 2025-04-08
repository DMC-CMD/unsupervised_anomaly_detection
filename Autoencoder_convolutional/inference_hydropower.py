import numpy as np

from autoencoder_convolutional import AutoencoderConvolutional
from evaluation_helper import display_roc, get_labels_and_reconstruction_errors, \
    get_auc_score, display_error_plot

autoencoder = AutoencoderConvolutional.load('Models/Hydropower/10')

normal_test = np.load('../Spectrograms/Hydropower/normal_test_spectrograms.npy')
anomaly_test = np.load('../Spectrograms/Hydropower/anomaly_test_spectrograms.npy')


normal_test_reconstructed = autoencoder.predict(normal_test, verbose=True)
anomaly_test_reconstructed = autoencoder.predict(anomaly_test, verbose=True)

labels, reconstruction_errors = get_labels_and_reconstruction_errors(normal_test, anomaly_test, normal_test_reconstructed, anomaly_test_reconstructed)
display_error_plot(labels, reconstruction_errors, 'Reconstruction error comparison', y_label='Reconstruction error')
display_roc(labels, reconstruction_errors, 'ROC convolutional AE with Hydropower data', 'convolutional Autoencoder')

auc_score = get_auc_score(labels, reconstruction_errors)
print('ROC-AUC: ', auc_score)


