import os
import timeit
import librosa
import sys
import numpy as np

from Preprocessing.features_helper import extract_features_from_file, feature_normalization, file_format_to_frame_format
from Preprocessing.spectrogram_helper import normalize_spectrograms, transform_to_input_shape, \
    zero_padding_time_bands_file


def extract_150_melspectrograms(input_dir):
    spectrograms = []
    for _, _, files in os.walk(input_dir):
        file_amount = 150
        counter = 0
        for file in sorted(files, key=lambda x: int(x.strip('.wav'))):
            counter += 1
            input_path = os.path.join(input_dir, file)
            signal, sr = librosa.load(input_path, sr=22050)
            melspectrogram = librosa.feature.melspectrogram(y=signal, n_fft=1024, hop_length=256, n_mels=64,
                                                            window='hann', win_length=1024)
            db_melspectrogram = librosa.power_to_db(melspectrogram)

            spectrograms.append(db_melspectrogram)

            sys.stdout.write(f'\r Folder {input_dir}. extracted melspectrogram from file {counter}/{file_amount}')
            if counter == file_amount:
                break

    print('')  # used to keep the last print from sys.stdout.write

    return spectrograms

def run_spectrogram_preprocessing(input_dir):
    melspectrograms = extract_150_melspectrograms(input_dir)
    melspectrograms = normalize_spectrograms(melspectrograms)
    melspectrograms = transform_to_input_shape(melspectrograms)
    np.save('dummy.npy', melspectrograms)

def run_mimii_spectrogram_preprocessing(input_dir):
    run_spectrogram_preprocessing(input_dir)
    zero_padding_time_bands_file('dummy.npy')

def extract_features_150_files(input_dir):
    features = np.array([[[]]])
    for root, _, files in os.walk(input_dir):
        file_amount = 150
        counter = 0
        for file in sorted(files, key=lambda x: int(x.strip('.wav'))):
            counter += 1
            full_path = os.path.join(root, file)
            current_features = extract_features_from_file(full_path)

            if features.shape[2] != current_features.shape[1]:  # initial case
                features = np.array([current_features])
            else:
                features = np.vstack((features, [current_features]))

            sys.stdout.write(f'\r Folder: {input_dir}, features extracted from file: {counter}/{file_amount}')
            if counter == file_amount:
                break

    print('')  # used to keep the last print from sys.stdout.write
    return features

def run_feature_preprocessing(input_dir):
    features = extract_features_150_files(input_dir)
    features = feature_normalization(features)
    features = file_format_to_frame_format(features)
    np.save('dummy.npy', features)


if __name__ == '__main__':
    executions = 20


    t = timeit.Timer(lambda: run_spectrogram_preprocessing('../HPP_dataset/normal'))
    r = t.repeat(executions, 1)
    print('Best runtime (s) to preprocess 150 Hydropower spectrograms: ', min(r))
    
    t = timeit.Timer(lambda: run_mimii_spectrogram_preprocessing('../MIMII_dataset/pump/id_00/normal'))
    r = t.repeat(executions, 1)
    print('Best runtime (s) to preprocess 150 MIMII spectrograms: ', min(r))

    t = timeit.Timer(lambda: run_feature_preprocessing('../HPP_dataset/normal'))
    r = t.repeat(executions, 1)
    print('Best runtime (s) to extract features from 150 Hydropower files: ', min(r))

    t = timeit.Timer(lambda: run_feature_preprocessing('../MIMII_dataset/pump/id_00/normal'))
    r = t.repeat(executions, 1)
    print('Best runtime (s) to extract features from 150 MIMII files: ', min(r))












