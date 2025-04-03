import librosa.feature
import numpy as np
import os
import sys

def extract_features_from_file(audio_path):
    signal, sr = librosa.load(audio_path, sr=22050)

    rms_energy = librosa.feature.rms(y=signal, frame_length=1024, hop_length=256)
    rms_energy = np.array(rms_energy[0])

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=1024, hop_length=256)
    zero_crossing_rate = np.array(zero_crossing_rate[0])

    stft = librosa.stft(signal, n_fft=1024, hop_length=256)
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

    melspectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(melspectrogram), sr=sr, n_mfcc=64)
    mfccs = np.array(mfccs)

    for mfcc in mfccs:
        feature_vectors = np.column_stack((mfcc, feature_vectors))
        
    return feature_vectors


def extract_features_from_folder(input_dir):

    features = np.array([[[]]])
    for root, _, files in os.walk(input_dir):
        file_amount = len(files)
        counter = 0
        for file in sorted(files, key=lambda x: int(x.strip('.wav'))):
            counter += 1
            full_path = os.path.join(root, file)
            current_features = extract_features_from_file(full_path)

            if features.shape[2] != current_features.shape[1]: #initial case
                features = np.array([current_features])
            else:
                features = np.vstack((features, [current_features]))

            sys.stdout.write(f'\r Folder: {input_dir}, features extracted from file: {counter}/{file_amount}')


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

def frame_format_to_file_format(frame_features, feature_values_per_file):
    file_features = []

    for i in range(0, len(frame_features), feature_values_per_file):
        file = frame_features[i:i+feature_values_per_file]
        file_features.append(file)

    return file_features


def save_to_file(path, data):
    if os.path.exists(path):
        os.remove(path)
    np.save(path, data)


