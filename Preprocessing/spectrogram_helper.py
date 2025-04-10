import librosa
import librosa.feature
import os
import numpy as np
import sys

from constants import SAMPLE_RATE, FRAME_SIZE, HOP_LENGTH, NUMBER_OF_MELS

def extract_melspectrograms(input_dir):
    spectrograms = []
    for _, _, files in os.walk(input_dir):
        number_of_files = len(files)
        counter = 0
        for file in sorted(files, key=lambda x: int(x.strip('.wav'))):
            counter += 1
            input_path = os.path.join(input_dir, file)
            signal, sr = librosa.load(input_path, sr=SAMPLE_RATE)
            melspectrogram = librosa.feature.melspectrogram(y=signal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mels=NUMBER_OF_MELS, window='hann', win_length=FRAME_SIZE)
            db_melspectrogram = librosa.power_to_db(melspectrogram)

            spectrograms.append(db_melspectrogram)

            sys.stdout.write(f'\r Folder {input_dir}. extracted melspectrogram from file {counter}/{number_of_files}')


    return spectrograms

def normalize_spectrograms(spectrograms):
    normalized_spectrograms = []
    for spectrogram in spectrograms:
        norm_melspectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        normalized_spectrograms.append(norm_melspectrogram)
    return np.array(normalized_spectrograms)

def transform_to_input_shape(spectrograms):
    spectrograms = np.array(spectrograms)
    spectrograms = spectrograms[..., np.newaxis]
    return spectrograms

def zero_padding_time_bands_file(file_path):
    spectrograms = np.load(file_path)
    spectrograms = spectrograms.tolist()
    number_of_spectrograms = len(spectrograms)
    number_of_frequency_bands = len(spectrograms[0])
    for i in range(0, number_of_spectrograms):
        for j in range(0, number_of_frequency_bands):
            spectrograms[i][j].append([0])
            spectrograms[i][j].append([0])

    spectrograms = np.array(spectrograms)
    np.save(file_path, spectrograms)

def transform_to_spectrogram(input_shaped_spectrograms):
    spectrograms = np.squeeze(input_shaped_spectrograms, axis=3)
    return spectrograms

def save_to_file(path, data):
    if os.path.exists(path):
        os.remove(path)
    np.save(path, data)

