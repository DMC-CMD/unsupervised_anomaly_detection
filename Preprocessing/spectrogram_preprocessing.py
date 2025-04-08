import os
from sklearn.model_selection import train_test_split

from Preprocessing.spectrogram_helper import extract_melspectrograms, normalize_spectrograms, transform_to_input_shape, \
    zero_padding_time_bands_file, save_to_file


def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def preprocess_normal_spectrograms(input_folder, output_folder):
    create_folder_if_not_exists(output_folder)
    normal_spectrograms = extract_melspectrograms(input_folder)
    normal_spectrograms = normalize_spectrograms(normal_spectrograms)
    normal_spectrograms = transform_to_input_shape(normal_spectrograms)

    normal_train, normal_rest = train_test_split(normal_spectrograms, test_size=0.3)
    normal_test, normal_validation = train_test_split(normal_rest, test_size=0.5)

    save_to_file(output_folder + '/normal_train_spectrograms.npy', normal_train)
    save_to_file(output_folder + '/normal_validation_spectrograms.npy', normal_validation)
    save_to_file(output_folder + '/normal_test_spectrograms.npy', normal_test)

def preprocess_anomaly_spectrograms(input_folder, output_folder):
    create_folder_if_not_exists(output_folder)
    anomaly_spectrograms = extract_melspectrograms(input_folder)
    anomaly_spectrograms = normalize_spectrograms(anomaly_spectrograms)
    anomaly_spectrograms = transform_to_input_shape(anomaly_spectrograms)


    anomaly_validation, anomaly_test = train_test_split(anomaly_spectrograms, test_size=0.5)

    save_to_file(output_folder + '/anomaly_validation_spectrograms.npy', anomaly_validation)
    save_to_file(output_folder + '/anomaly_test_spectrograms.npy', anomaly_test)

def zero_padding_frequency_bands(folder):
    for _, _, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(folder, file)
            zero_padding_time_bands_file(full_path)