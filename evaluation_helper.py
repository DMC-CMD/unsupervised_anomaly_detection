from sklearn.metrics import roc_curve, auc, RocCurveDisplay, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter


def display_training_process(history, title):
    plt.plot(history.history['loss'], c='b', label='Training data')
    plt.plot(history.history['val_loss'], color='orange', label='Validation data')
    plt.title(title)
    plt.ylabel('Reconstruction error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show(dpi=300)

def get_labels_and_reconstruction_errors(normal_spectrograms, anomaly_spectrograms, reconstructed_normals, reconstructed_anomalies):
    print('Normal spectrogram shape before: ', normal_spectrograms.shape)
    normal_spectrograms = transform_to_spectrogram(normal_spectrograms)
    print('Normal spectrogram shape after: ', normal_spectrograms.shape)

    anomaly_spectrograms= transform_to_spectrogram(anomaly_spectrograms)
    print('Reconstructed normal shape before: ', reconstructed_normals.shape)
    reconstructed_normals = transform_to_spectrogram(reconstructed_normals)
    print('Reconstructed normal shape after: ', reconstructed_normals.shape)
    reconstructed_anomalies = transform_to_spectrogram(reconstructed_anomalies)

    reconstruction_errors = []

    for i in range(len(normal_spectrograms)):
        current_error = mean_squared_error(normal_spectrograms[i], reconstructed_normals[i])
        reconstruction_errors.append(current_error)

    for i in range(len(anomaly_spectrograms)):
        current_error = mean_squared_error(anomaly_spectrograms[i], reconstructed_anomalies[i])
        reconstruction_errors.append(current_error)

    labels_normal = [0] * len(normal_spectrograms)
    labels_anomaly = [1] * len(anomaly_spectrograms)
    labels = np.concatenate((labels_normal, labels_anomaly))
    return labels, reconstruction_errors


# the autoencoder outputs are in the form of a grayscale image (x, y, 1),
# in order to display it as a spectrogram it is transformed back to (x, y)
def transform_to_spectrogram(input_shaped_spectrograms):
    spectrograms = np.squeeze(input_shaped_spectrograms, axis=3)
    return spectrograms

def get_labels_and_quantization_errors(normal_features, anomaly_features, som):
    normal_quantization_errors = []
    for sample in normal_features:
        current_error = som.quantization_error(sample)
        normal_quantization_errors.append(current_error)

    anomaly_quantization_errors = []
    for sample in anomaly_features:
        current_error = som.quantization_error(sample)
        anomaly_quantization_errors.append(current_error)

    quantization_errors = np.concatenate((normal_quantization_errors, anomaly_quantization_errors))

    labels_normal = [0] * len(normal_features)
    labels_anomaly = [1] * len(anomaly_features)
    labels = np.concatenate((labels_normal, labels_anomaly))

    return labels, quantization_errors

def display_roc(true_labels, predicted_score, title, model_name):
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_score)
    roc_auc = auc(fpr, tpr)

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                              estimator_name=model_name)
    display.plot()
    plt.title(title)
    plt.show()

def display_error_plot(true_labels, predicted_scores, title, y_label, normal_color='g', anomaly_color='r', ):
    scores_sorted, labels_sorted = sort_labeled_data(true_labels, predicted_scores)
    colors = [normal_color if x == 0 else anomaly_color for x in labels_sorted]

    x = np.arange(len(scores_sorted))
    fig, ax = plt.subplots()
    normal_patch = mpatches.Patch(color=normal_color, label='Normal')
    anomaly_patch = mpatches.Patch(color=anomaly_color, label='Anomaly')
    plt.scatter(x, scores_sorted, c=colors, s=3)
    ax.legend(handles=[normal_patch, anomaly_patch])

    plt.title(title)
    plt.xlabel('Sample Number')
    plt.ylabel(y_label)
    plt.show()

def frame_format_to_file_format(frame_features, feature_values_per_file):
    file_features = []

    for i in range(0, len(frame_features), feature_values_per_file):
        file = frame_features[i:i+feature_values_per_file]
        file_features.append(file)

    return np.array(file_features)

def sort_labeled_data(true_labels, predicted_scores):
    labels_sorted = [x for _, x in sorted(zip(predicted_scores, true_labels))]
    scores_sorted = sorted(predicted_scores)
    return scores_sorted, labels_sorted

def get_auc_score(true_labels, predicted_score):
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_score)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def display_confusion_matrix(true_labels, predicted_scores, threshold, title):
    predictions = [1 if predicted_score >= threshold else 0 for predicted_score in predicted_scores]
    conf_matrix = confusion_matrix(true_labels, predictions)
    img =ConfusionMatrixDisplay(conf_matrix, display_labels=['Normal', 'Anomaly'])
    img.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show(dpi=300)

def display_hit_map(som, sidelenght, train_frames, title):
    bmu_list = [som.winner(x) for x in train_frames]
    bmu_counts = Counter(bmu_list)

    hit_map = np.zeros((sidelenght, sidelenght))
    for (x, y), count in bmu_counts.items():
        hit_map[x, y] = count

    plt.figure(figsize=(8, 8))
    plt.pcolor(hit_map, cmap='Blues')
    plt.colorbar(label="BMU Hits")
    plt.title(title)
    plt.show()

def display_distance_map(som, title):
    plt.figure(figsize=(8, 8))
    plt.pcolor(som.distance_map().T, cmap='gist_yarg')
    plt.title(title)
    plt.colorbar()
    plt.show()