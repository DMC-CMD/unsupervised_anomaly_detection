import numpy as np
import pickle

from Preprocessing.features_helper import frame_format_to_file_format
from evaluation_helper import get_labels_and_quantization_errors, display_roc, display_reconstruction_errors, \
    get_auc_score, get_minimal_anomaly_score, display_confusion_matrix

with open('Models/Hydropower/base_model.p', 'rb') as file:
    som = pickle.load(file)

normal_test_features = np.load('../Features/Hydropower/normal_test_features.npy')
normal_test_samples = frame_format_to_file_format(normal_test_features, 64)
anomaly_test_features = np.load('../Features/Hydropower/anomaly_test_features.npy')
anomaly_test_samples = frame_format_to_file_format(anomaly_test_features, 64)

labels, quantization_errors = get_labels_and_quantization_errors(normal_test_samples, anomaly_test_samples, som)
display_reconstruction_errors(labels, quantization_errors, 'Quantization errors for SOM with Hydropower data')
display_roc(labels, quantization_errors, 'ROC for SOM with Hydropower data', 'Self-Organizing Map')

auc_score = get_auc_score(labels, quantization_errors)
print(auc_score)

minimal_anomaly_score = get_minimal_anomaly_score(labels, quantization_errors)
display_confusion_matrix(labels, quantization_errors, threshold=minimal_anomaly_score, title='Confusion matrix')



