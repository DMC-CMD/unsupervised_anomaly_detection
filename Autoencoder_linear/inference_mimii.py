import numpy as np

from autoencoder_linear import AutoencoderLinear
from evaluation_helper import get_labels_and_reconstruction_errors, display_reconstruction_errors, display_roc, \
    get_auc_score, get_minimal_anomaly_score, display_confusion_matrix

autoencoder = AutoencoderLinear.load('Models/MIMII/base_model')

normal_test = np.load('../Spectrograms/MIMII/normal_test_spectrograms.npy')
anomaly_test = np.load('../Spectrograms/MIMII/anomaly_test_spectrograms.npy')

reconstructed_normal = autoencoder.predict(normal_test)
reconstructed_anomalies = autoencoder.predict(anomaly_test)

labels, reconstruction_errors = get_labels_and_reconstruction_errors(normal_test, anomaly_test, reconstructed_normal, reconstructed_anomalies)
display_reconstruction_errors(labels, reconstruction_errors, 'Reconstruction errors linear AE with MIMII data')
display_roc(labels, reconstruction_errors, 'Receiver Operating Characteristic', 'linear Autoencoder')

auc_score = get_auc_score(labels, reconstruction_errors)
print(auc_score)

minimal_anomaly_score = get_minimal_anomaly_score(labels, reconstruction_errors)
display_confusion_matrix(labels, reconstruction_errors, threshold=minimal_anomaly_score, title='Confusion matrix')


