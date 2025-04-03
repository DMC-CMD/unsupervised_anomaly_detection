import numpy as np

from autoencoder_linear import AutoencoderLinear
from evaluation_helper import get_labels_and_reconstruction_errors, display_reconstruction_errors, display_roc, \
    get_auc_score, get_minimal_anomaly_score, display_confusion_matrix

autoencoder = AutoencoderLinear.load('Models/Hydropower/base_model')

normal_test = np.load('../Spectrograms/Hydropower/normal_test_spectrograms.npy')
anomaly_test = np.load('../Spectrograms/Hydropower/anomaly_test_spectrograms.npy')

reconstructed_normal = autoencoder.predict(normal_test, verbose=True)
reconstructed_anomaly = autoencoder.predict(anomaly_test, verbose=True)

labels, reconstruction_errors = get_labels_and_reconstruction_errors(normal_test, anomaly_test, reconstructed_normal, reconstructed_anomaly)
display_reconstruction_errors(labels, reconstruction_errors, 'Reconstruction errors linear AE with Hydropower data')
display_roc(labels, reconstruction_errors, 'ROC linear AE with Hydropower data', 'linear Autoencoder')

auc_score = get_auc_score(labels, reconstruction_errors)
print(auc_score)

minimal_anomaly_score = get_minimal_anomaly_score(labels, reconstruction_errors)
display_confusion_matrix(labels, reconstruction_errors, threshold=minimal_anomaly_score, title='Confusion matrix')



