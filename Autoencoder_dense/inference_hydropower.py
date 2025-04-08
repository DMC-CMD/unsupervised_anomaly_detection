import numpy as np

from autoencoder_dense import AutoencoderDense
from evaluation_helper import get_labels_and_reconstruction_errors, display_roc, \
    get_auc_score, display_error_plot


autoencoder = AutoencoderDense.load('Models/Hydropower/base_model')

normal_test = np.load('../Spectrograms/Hydropower/normal_test_spectrograms.npy')
anomaly_test = np.load('../Spectrograms/Hydropower/anomaly_test_spectrograms.npy')

reconstructed_normal = autoencoder.predict(normal_test, verbose=True)
reconstructed_anomaly = autoencoder.predict(anomaly_test, verbose=True)

labels, reconstruction_errors = get_labels_and_reconstruction_errors(normal_test, anomaly_test, reconstructed_normal, reconstructed_anomaly)
display_error_plot(labels, reconstruction_errors, 'Reconstruction errors dense AE with Hydropower data', y_label='Reconstruction error')
display_roc(labels, reconstruction_errors, 'ROC dense AE with Hydropower data', 'dense Autoencoder')

auc_score = get_auc_score(labels, reconstruction_errors)
print('ROC-AUC: ', auc_score)




