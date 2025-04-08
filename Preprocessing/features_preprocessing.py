import os
from sklearn.model_selection import train_test_split

from features_helper import extract_features_from_folder, file_format_to_frame_format, feature_normalization, save_to_file


def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def preprocess_normal_features(input_folder, output_folder):
    create_folder_if_not_exists(output_folder)

    #output paths
    normal_train_path = output_folder + '/normal_train_features.npy'
    normal_test_path = output_folder + '/normal_test_features.npy'
    normal_validation_path = output_folder + '/normal_validation_features.npy'


    normal_features = extract_features_from_folder(input_folder)
    normal_features = feature_normalization(normal_features)
    normal_train_features, normal_rest_features = train_test_split(normal_features, test_size=0.3)
    normal_test_features, normal_validation_features = train_test_split(normal_rest_features, test_size=0.5)

    normal_train_frames = file_format_to_frame_format(normal_train_features)
    normal_test_frames = file_format_to_frame_format(normal_test_features)
    normal_validation_frames = file_format_to_frame_format(normal_validation_features)

    save_to_file(normal_train_path, normal_train_frames)
    save_to_file(normal_test_path, normal_test_frames)
    save_to_file(normal_validation_path, normal_validation_frames)


def preprocess_anomaly_features(input_folder, output_folder):
    create_folder_if_not_exists(output_folder)

    anomaly_validation_path = output_folder + '/anomaly_validation_features.npy'
    anomaly_test_path = output_folder + '/anomaly_test_features.npy'

    anomaly_features = extract_features_from_folder(input_folder)
    anomaly_features = feature_normalization(anomaly_features)

    anomaly_validation_features, anomaly_test_features = train_test_split(anomaly_features, test_size=0.5)
    print('Normal train features shape after:', anomaly_validation_features.shape)
    anomaly_validation_features = file_format_to_frame_format(anomaly_validation_features)
    print('Normal train features shape after:', anomaly_validation_features.shape)

    anomaly_test_features = file_format_to_frame_format(anomaly_test_features)

    save_to_file(anomaly_validation_path, anomaly_validation_features)
    save_to_file(anomaly_test_path, anomaly_test_features)

