import random
import sys
import os
import csv
import pickle
import math

from implementation import train_som, save_som
from parameter_tuning_helper import get_auc_validation_score_for_som


def generate_random_parameters(parameter_ranges):
    sidelength = random.randrange(parameter_ranges['sidelength'][0], parameter_ranges['sidelength'][1])

    sigma = random.randrange(parameter_ranges['sigma'][0], parameter_ranges['sigma'][1])
    max_sigma = math.floor(math.sqrt(sidelength**2 + sidelength**2)) # upper bound based on the map size to prevent sigma that overflows the map
    sigma = min(sigma, max_sigma)
    num_epochs = random.randrange(parameter_ranges['num_epochs'][0], parameter_ranges['num_epochs'][1])
    learning_rate = random.randrange(int(parameter_ranges['learning_rate'][0]*10), int(parameter_ranges['learning_rate'][1]*10)) / 10

    return sidelength, sigma, num_epochs, learning_rate

#integrates generation of report directly, since MiniSOM doesn't allow to
def random_parameter_tuning(parameter_ranges, rounds, dataset):

    for round in range(rounds):
        sys.stdout.write(f'\r Tuning round: {round+1}/{rounds}')

        sidelength, sigma, num_epochs, learning_rate = generate_random_parameters(parameter_ranges)
        som = train_som(sidelength, sigma, learning_rate, num_epochs, dataset=dataset)
        save_som(som, sidelength, sigma, num_epochs, learning_rate, f'Models/{dataset}/{round}')

    print('')  # used to keep the last print from sys.stdout.write


def create_models_report(models_folder, report_file_path):
    if os.path.exists(report_file_path):
        os.remove(report_file_path)

    dataset = models_folder.replace('Models/', '')

    with open(report_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'sidelength', 'sigma', 'num_epochs', 'learning_rate', 'validation roc_auc_score'])

        for _, dirs, _ in os.walk(models_folder):
            counter = 0
            max_count = len(dirs)
            for dir in sorted(dirs):
                counter += 1
                sys.stdout.write(f'\r Creating {dataset} Model Report: {counter}/{max_count}')
                full_path = os.path.join(models_folder, dir)

                with open(full_path + '/model.pkl', 'rb') as file:
                    current_model = pickle.load(file)

                with open(full_path + '/parameters.pkl', 'rb') as file:
                    current_parameters = pickle.load(file)
                sidelength = current_parameters[0]
                sigma = current_parameters[1]
                learning_rate = current_parameters[2]
                num_epochs = current_parameters[3]

                auc_score = get_auc_validation_score_for_som(current_model, dataset)

                writer.writerow([dir, sidelength, sigma, learning_rate, num_epochs, auc_score])
            print('') # used to keep the last print from sys.stdout.write

if __name__ == '__main__':
    parameter_ranges = {
        'sidelength': [8, 20],
        'sigma': [1, 20],
        'learning_rate': [0.1, 1.5],
        'num_epochs': [2, 10]
    }

    rounds = 100

    random_parameter_tuning(parameter_ranges, rounds,'Hydropower')
    create_models_report('Models/Hydropower', 'Models_report_hydropower_som.csv')

    random_parameter_tuning(parameter_ranges, rounds, 'MIMII')
    create_models_report('Models/MIMII', 'Models_report_mimii_som.csv')



