import numpy as np
import pickle

from constants import MIMII_FEATURE_VALUES_PER_FILE
from evaluation_helper import get_labels_and_quantization_errors, display_roc, \
    get_auc_score, display_error_plot, frame_format_to_file_format, display_distance_map

with open('Models/MIMII/base_model/model.pkl', 'rb') as file:
    som = pickle.load(file)

normal_test_features = np.load('../Features/MIMII/normal_test_features.npy')
normal_test_samples = frame_format_to_file_format(normal_test_features, MIMII_FEATURE_VALUES_PER_FILE)
anomaly_test_features = np.load('../Features/MIMII/anomaly_test_features.npy')
anomaly_test_samples = frame_format_to_file_format(anomaly_test_features, MIMII_FEATURE_VALUES_PER_FILE)

display_distance_map(som, 'Distance Map for MIMII model')

labels, quantization_errors = get_labels_and_quantization_errors(normal_test_samples, anomaly_test_samples, som)
display_error_plot(labels, quantization_errors, 'Quantization errors for SOM with MIMII data', y_label='Quantization error')
display_roc(labels, quantization_errors, 'ROC for SOM with MIMII data', 'Self-Organizing Map')

auc_score = get_auc_score(labels, quantization_errors)
print('ROC-AUC: ', auc_score)