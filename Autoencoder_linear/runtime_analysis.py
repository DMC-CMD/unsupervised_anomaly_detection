import timeit
import numpy as np
from sklearn.metrics import mean_squared_error


def run_inference(input_file, autoencoder):
    normal_test = np.load(input_file)
    normal_test = normal_test[:150]

    reconstructed_normals = autoencoder.predict(normal_test)
    threshold = 0.0016 # for the runtime evaluation, the specific threshold is not relevant and is therefore fixed
    classifications = []
    for i in range(len(reconstructed_normals)):
        current_error = mean_squared_error(normal_test[i], reconstructed_normals[i])
        if current_error >= threshold:
            classifications.append(1)
        else:
            classifications.append(0)

if __name__ == "__main__":
    # load the autoencoder you want to use for the runtime analysis
    executions = 1

    total_inference_time_hp = timeit.timeit(lambda : run_inference('../Spectrograms/Hydropower/normal_test_spectrograms.npy', 10), number=1)
    inference_time_per_execution_hp = total_inference_time_hp / executions
    print("Inference time (s) per execution for 150 Hydropower spectrograms: ", inference_time_per_execution_hp)

    total_inference_time_mimii = timeit.timeit(lambda: run_inference('../Spectrograms/MIMII/normal_test_spectrograms.npy', 10), number=1)
    inference_time_per_execution_mimii = total_inference_time_mimii / executions
    print("Inference time (s) per execution for 150 MIMII spectrograms: ", inference_time_per_execution_mimii)


