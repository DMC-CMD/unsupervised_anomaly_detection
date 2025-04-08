import numpy as np
import pickle

from evaluation_helper import get_labels_and_quantization_errors, display_roc, display_error_plot, \
    get_auc_score, frame_format_to_file_format, display_distance_map

with open('Models/Hydropower/base_model/model.pkl', 'rb') as file:
    som = pickle.load(file)

normal_test_features = np.load('../Features/Hydropower/normal_test_features.npy')
normal_test_samples = frame_format_to_file_format(normal_test_features, 64)
anomaly_test_features = np.load('../Features/Hydropower/anomaly_test_features.npy')
anomaly_test_samples = frame_format_to_file_format(anomaly_test_features, 64)

display_distance_map(som, 'Distance Map Hydropower model')

labels, quantization_errors = get_labels_and_quantization_errors(normal_test_samples, anomaly_test_samples, som)
display_error_plot(labels, quantization_errors, 'Quantization error comparison', y_label='Quantization error')
display_roc(labels, quantization_errors, 'ROC for SOM with Hydropower data', 'Self-Organizing Map')

auc_score = get_auc_score(labels, quantization_errors)
print('ROC-AUC: ', auc_score)



