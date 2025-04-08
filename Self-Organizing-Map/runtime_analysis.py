import timeit
import numpy as np
import pickle

from evaluation_helper import frame_format_to_file_format

def run_inference(input_file, som, feature_values_per_file):
    normal_test_features = np.load(input_file)
    normal_test_features = normal_test_features[:150*feature_values_per_file]
    normal_test_files = frame_format_to_file_format(normal_test_features, feature_values_per_file)

    threshold = 0.0016  # for the runtime evaluation, the specific threshold is not relevant and is therefore fixed
    classifications = []

    for sample in normal_test_files:
        current_error = som.quantization_error(sample)
        if current_error >= threshold:
            classifications.append(1)
        else:
            classifications.append(0)


if __name__ == "__main__":
    # load the specific autoencoders you want to use for the runtime analysis
    executions = 20

    with open('Models/Hydropower/base_model/model.pkl', 'rb') as file:
        tuned_model_hydropower = pickle.load(file)
    t = timeit.Timer(lambda : run_inference('../Features/Hydropower/normal_test_features.npy', tuned_model_hydropower, 64))
    r = t.repeat(executions, 1)
    print('Best runtime (s) to classify 150 Hydropower files with Self-Organizing Map : ', min(r))


    with open('Models/MIMII/5/model.pkl', 'rb') as file:
        tuned_model_mimii = pickle.load(file)
    t = timeit.Timer(lambda: run_inference('../Features/MIMII/normal_test_features.npy', tuned_model_mimii, 862))
    r = t.repeat(executions, 1)
    print('Best runtime (s) to classify 150 MIMII files with Self-Organizing Map: ', min(r))



