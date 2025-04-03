import numpy as np

from autoencoder_convolutional import AutoencoderConvolutional
from evaluation_helper import display_reconstruction_errors, display_roc, get_labels_and_reconstruction_errors, \
    get_auc_score, display_confusion_matrix, get_minimal_anomaly_score

autoencoder = AutoencoderConvolutional.load('Models/Hydropower/base_model')

normal_test = np.load('../Spectrograms/Hydropower/normal_test_spectrograms.npy')
anomaly_test = np.load('../Spectrograms/Hydropower/anomaly_test_spectrograms.npy')


normal_test_reconstructed = autoencoder.predict(normal_test, verbose=True)
anomaly_test_reconstructed = autoencoder.predict(anomaly_test, verbose=True)

labels, reconstruction_errors = get_labels_and_reconstruction_errors(normal_test, anomaly_test, normal_test_reconstructed, anomaly_test_reconstructed)
display_reconstruction_errors(labels, reconstruction_errors, 'Reconstruction error comparison')
display_roc(labels, reconstruction_errors, 'ROC convolutional AE with Hydropower data', 'convolutional Autoencoder')

auc_score = get_auc_score(labels, reconstruction_errors)
print(auc_score)

minimal_anomaly_score = get_minimal_anomaly_score(labels, reconstruction_errors)
display_confusion_matrix(labels, reconstruction_errors, threshold=minimal_anomaly_score, title='Confusion matrix')

