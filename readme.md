# Acoustic Anomaly Detection in Hydropower Plants


## Introduction
This GitHub repository provides the source code to the Bachelor Thesis written at the University of Zurich (UZH) under supervision of Prof.Dr. Bruno Rodrigues, Jan van der Assen and Prof. Dr. Burkhard Stiller.


## Required packages
The project needs the following dependencies to be run:
- Python 3.9 or later
- Librosa
- Numpy
- Scikit-learn
- Matplotlib
- MiniSom
- Tensorflow

## Datasets
This source code uses two datasets to train and evaluate the implemented models. 
The self-created hydropower dataset contains normal and anomalous sound samples stored in the HPP_dataset directory.
Some of the following code files require additionally the MIMII dataset to be available in a folder called 'MIMII_dataset'. It was too big to be included in the repository due to GitHub's size limit of 2GB. It can be downloaded and included in a local repository: {TODO: INSERT LINK }

## 1. Preprocessing
- 'preprocessing.py' extracts features vectors and spectrograms from both datasets and stores them in the directories 'Features' and 'Spectrograms'

Run it to make the preprocessed files available for training and evaluation.

## 2. Models

Model types:
- Dense Autoencoder
- Convolutional Autoencoder
- Self-Organizing Map

For each model type, there is a directory containing the following files:
- 'implementation.py' trains and saves two models in their base configuration, each model is tied to one of the datasets.
- 'inference_hydropower.py' creates evaluation plots and prints the ROC_AUC score for the hydropower base model.
- 'inference_mimii.py' the same for the MIMII base model
- 'parameter_tuning.py' runs hyperparameter tuning for the number of rounds specified within the specified parameter ranges and saves the models in the directory 'Models'. 

## 3. Runtime Analysis
Each model type has a 'runtime_analysis.py' file to measure the inference runtime for the specified models using 150 spectrogram/feature samples.
Additionally, the preprocessing directory has another 'runtime_analysis.py' file to measure the runtime to preprocess 150 audio samples. 




