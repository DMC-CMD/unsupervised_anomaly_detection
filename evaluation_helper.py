from sklearn.metrics import roc_curve, auc, RocCurveDisplay, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def display_training_process(history, title):
    plt.plot(history.history['loss'], c='b', label='Training data')
    plt.plot(history.history['val_loss'], color='orange', label='Validation data')
    plt.title(title)
    plt.ylabel('Reconstruction error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show(dpi=300)

def get_labels_and_reconstruction_errors(normal_spectrograms, anomaly_spectrograms, reconstructed_normals, reconstructed_anomalies):
    normal_spectrograms = transform_to_spectrogram(normal_spectrograms)
    anomaly_spectrograms= transform_to_spectrogram(anomaly_spectrograms)
    reconstructed_normals = transform_to_spectrogram(reconstructed_normals)
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
    plt.savefig('roc_lae_mimii', dpi=300)

def get_minimal_anomaly_score(true_labels, predicted_score):
    min_score = np.min(predicted_score)
    for i in range(len(true_labels)):
        if true_labels[i] == 1:
            if predicted_score[i] < min_score:
                min_score = predicted_score[i]
    return min_score

def display_reconstruction_errors(true_labels, predicted_scores, title, normal_color='g', anomaly_color='r', ):
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
    plt.ylabel('Reconstruction error')
    plt.savefig('reconstruction_errors_lae_hydropower', dpi=300)



def sort_labeled_data(true_labels, predicted_scores):
    labels_sorted = [x for _, x in sorted(zip(predicted_scores, true_labels))]
    scores_sorted = sorted(predicted_scores)
    return scores_sorted, labels_sorted

def get_auc_score(true_labels, predicted_score):
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_score)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def get_minimal_anomaly_score(true_labels, predicted_scores, anomaly_label=1):
    scores_sorted, labels_sorted = sort_labeled_data(true_labels, predicted_scores)
    for i in range(len(scores_sorted)):
        if labels_sorted[i] == anomaly_label:
            return scores_sorted[i]

def display_confusion_matrix(true_labels, predicted_scores, threshold, title):
    predictions = [1 if predicted_score >= threshold else 0 for predicted_score in predicted_scores]
    conf_matrix = confusion_matrix(true_labels, predictions)
    img =ConfusionMatrixDisplay(conf_matrix, display_labels=['Normal', 'Anomaly'])
    img.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show(dpi=300)



