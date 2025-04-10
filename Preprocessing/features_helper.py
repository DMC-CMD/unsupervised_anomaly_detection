import librosa.feature
import numpy as np
import os
import sys
from constants import SAMPLE_RATE, HOP_LENGTH, FRAME_SIZE, NUMBER_OF_MELS


def extract_features_from_file(audio_path):
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    rms_energy = librosa.feature.rms(y=signal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)
    rms_energy = np.array(rms_energy[0])

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)
    zero_crossing_rate = np.array(zero_crossing_rate[0])

    stft = librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    stft = np.abs(stft)

    spectral_centroid = librosa.feature.spectral_centroid(S=stft)
    spectral_centroid = np.array(spectral_centroid[0])

    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft)
    spectral_bandwidth = np.array(spectral_bandwidth[0])

    spectral_contrast = librosa.feature.spectral_contrast(S=stft)
    spectral_contrast = np.array(spectral_contrast[0])

    spectral_flatness = librosa.feature.spectral_flatness(S=stft)
    spectral_flatness = np.array(spectral_flatness[0])

    spectral_rolloff = librosa.feature.spectral_rolloff(S=stft)
    spectral_rolloff = np.array(spectral_rolloff[0])

    feature_vectors = np.column_stack((rms_energy, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, zero_crossing_rate))

    melspectrogram = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mels=NUMBER_OF_MELS)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(melspectrogram), sr=SAMPLE_RATE, n_mfcc=NUMBER_OF_MELS)
    mfccs = np.array(mfccs)

    for mfcc in mfccs:
        feature_vectors = np.column_stack((mfcc, feature_vectors))
        
    return feature_vectors


def extract_features_from_folder(input_dir):

    features = np.array([[[]]])
    for root, _, files in os.walk(input_dir):
        number_of_files = len(files)
        counter = 0
        for file in sorted(files, key=lambda x: int(x.strip('.wav'))):
            counter += 1
            full_path = os.path.join(root, file)
            current_features = extract_features_from_file(full_path)

            if features.shape[2] != current_features.shape[1]: #initial case
                features = np.array([current_features])
            else:
                features = np.vstack((features, [current_features]))

            sys.stdout.write(f'\r Folder: {input_dir}, features extracted from file: {counter}/{number_of_files}')


    return features


def feature_normalization(feature_vectors):
    vectors_t = feature_vectors.T
    for i in range(len(vectors_t)):
        vectors_t[i] = (vectors_t[i] - vectors_t[i].min()) / (vectors_t[i].max() - vectors_t[i].min())

    return vectors_t.T


def file_format_to_frame_format(file_features):
    frame_features = []
    for i in range(len(file_features)):
        for j in range(len(file_features[i])):
            frame_features.append(file_features[i][j])

    frame_features = np.array(frame_features)

    return frame_features


def save_to_file(path, data):
    if os.path.exists(path):
        os.remove(path)
    np.save(path, data)


