from Preprocessing.spectrogram_preprocessing import preprocess_normal_spectrograms, \
    preprocess_anomaly_spectrograms, zero_padding_frequency_bands
from features_preprocessing import preprocess_normal_features, preprocess_anomaly_features

import numpy as np


hydropower_input_folder = '../HPP_dataset'

hydropower_features_output_folder = '../Features/Hydropower'
preprocess_normal_features(hydropower_input_folder + '/normal', hydropower_features_output_folder)
preprocess_anomaly_features(hydropower_input_folder + '/abnormal', hydropower_features_output_folder)

hydropower_spectrograms_output_folder = '../Spectrograms/Hydropower'
preprocess_normal_spectrograms(hydropower_input_folder + '/normal', hydropower_spectrograms_output_folder)
preprocess_anomaly_spectrograms(hydropower_input_folder + '/abnormal', hydropower_spectrograms_output_folder)



mimii_input_folder = '../MIMII_dataset/pump/id_00'

mimii_features_output_folder = '../Features/MIMII'
preprocess_normal_features(mimii_input_folder + '/normal', mimii_features_output_folder)
preprocess_anomaly_features(mimii_input_folder + '/abnormal', mimii_features_output_folder)


mimii_spectrograms_output_folder = '../Spectrograms/MIMII'
preprocess_normal_spectrograms(mimii_input_folder + '/normal', mimii_spectrograms_output_folder)
preprocess_anomaly_spectrograms(mimii_input_folder + '/abnormal', mimii_spectrograms_output_folder)
zero_padding_frequency_bands(mimii_spectrograms_output_folder)
