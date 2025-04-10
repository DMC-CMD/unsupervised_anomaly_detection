import os
import random
import csv
import sys

from Autoencoder_dense.autoencoder_dense import AutoencoderDense
from parameter_tuning_helper import get_auc_validation_score_for_ae, random_power_of_two
from implementation import train_dense_ae


def generate_random_parameters(parameter_ranges):
    first_layer_output_size = random_power_of_two(parameter_ranges['first_layer_output_size'][0], parameter_ranges['first_layer_output_size'][1])

    max_bottleneck_size = int(first_layer_output_size /8)
    bottleneck_size = random_power_of_two(parameter_ranges['bottleneck_size'][0],
                                          parameter_ranges['bottleneck_size'][1])
    bottleneck_size = min(bottleneck_size, max_bottleneck_size)

    batch_size = random_power_of_two(parameter_ranges['batch_size'][0], parameter_ranges['batch_size'][1])
    epochs = random.randrange(parameter_ranges['epochs'][0], parameter_ranges['epochs'][1], 5)

    return first_layer_output_size, bottleneck_size, batch_size, epochs

def random_parameter_tuning(parameter_ranges, rounds, dataset):
    for round in range(rounds):
        sys.stdout.write(f'\r Tuning {dataset} round: {round+1}/{rounds}')

        first_layer_output_size, bottleneck_size, batch_size, epochs = generate_random_parameters(parameter_ranges)
        autoencoder = train_dense_ae(first_layer_output_size, bottleneck_size, batch_size, epochs, dataset=dataset, verbose=False)
        autoencoder.save(f'Models/{dataset}/' + str(round))
    print('')  # used to keep the last print from sys.stdout.write


def create_models_report(models_folder, report_file_path):
    if os.path.exists(report_file_path):
        os.remove(report_file_path)

    dataset = models_folder.replace('Models/', '')

    with open(report_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'first_layer_output_size', 'bottleneck_size', 'batch_size', 'epochs', 'validation roc_auc_score'])

        for _, dirs, _ in os.walk(models_folder):
            counter = 0
            max_count = len(dirs)
            for dir in sorted(dirs):
                counter += 1
                sys.stdout.write(f'\r Creating {dataset} Model Report: {counter}/{max_count}')
                full_path = os.path.join(models_folder, dir)

                current_model = AutoencoderDense.load(full_path)

                layer_output_sizes, bottleneck_size = current_model.get_hyperparameters()
                batch_size, epochs = current_model.get_train_parameters()
                auc_score = get_auc_validation_score_for_ae(current_model, dataset)

                writer.writerow([dir, layer_output_sizes[0], bottleneck_size, batch_size, epochs, auc_score])
            print('') # used to keep the last print from sys.stdout.write


if __name__ == '__main__':

    parameter_ranges = {
        'first_layer_output_size': [64, 512],
        'bottleneck_size': [4, 128],

        'batch_size': [8, 128],
        'epochs': [5, 100]
    }

    rounds = 100

    random_parameter_tuning(parameter_ranges, rounds, dataset='Hydropower')
    create_models_report('Models/Hydropower', 'Models_report_hydropower_dense_ae.csv')

    random_parameter_tuning(parameter_ranges, rounds, dataset='MIMII')
    create_models_report('Models/MIMII', 'Models_report_mimii_dense_ae.csv')




