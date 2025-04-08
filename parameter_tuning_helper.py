import numpy as np
import random
import math

from evaluation_helper import get_labels_and_reconstruction_errors, get_auc_score, get_labels_and_quantization_errors, \
    frame_format_to_file_format


def get_auc_validation_score_for_ae(autoencoder, dataset):
    normal_test = np.load(f'../Spectrograms/{dataset}/normal_validation_spectrograms.npy')
    anomaly_test = np.load(f'../Spectrograms/{dataset}/anomaly_validation_spectrograms.npy')

    reconstructed_normal = autoencoder.predict(normal_test, verbose=False)
    reconstructed_anomaly = autoencoder.predict(anomaly_test, verbose=False)

    labels, reconstruction_errors = get_labels_and_reconstruction_errors(normal_test, anomaly_test, reconstructed_normal, reconstructed_anomaly)
    auc_score = get_auc_score(labels, reconstruction_errors)
    return auc_score

def get_auc_validation_score_for_som(som, dataset):
    normal_test = np.load(f'../Features/{dataset}/normal_validation_features.npy')
    anomaly_test = np.load(f'../Features/{dataset}/anomaly_validation_features.npy')

    feature_values_per_file = 0
    if dataset == 'Hydropower':
        feature_values_per_file = 64
    if dataset == 'MIMII':
        feature_values_per_file = 862

    normal_test = frame_format_to_file_format(normal_test, feature_values_per_file)
    anomaly_test = frame_format_to_file_format(anomaly_test, feature_values_per_file)

    labels, quantization_errors = get_labels_and_quantization_errors(normal_test, anomaly_test, som)

    auc_score = get_auc_score(labels, quantization_errors)
    return auc_score


def random_power_of_two(start, end):
    x = random.randint(int(math.log2(start)), int(math.log2(end)))
    return int(2** x)