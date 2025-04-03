from minisom import MiniSom
import numpy as np
import pickle
import os


def train_som(sidelength, sigma, learning_rate, num_epochs, dataset, verbose=False):
    normal_train = np.load(f'../Features/{dataset}/normal_train_features.npy')
    feature_amount = normal_train.shape[1]

    som = MiniSom(
        topology='rectangular',
        x=sidelength,
        y=sidelength,
        input_len=feature_amount,
        neighborhood_function='gaussian',
        sigma=sigma,
        sigma_decay_function='linear_decay_to_one',
        activation_distance='euclidean',
        learning_rate=learning_rate,
        decay_function='asymptotic_decay'
    )

    som.random_weights_init(normal_train)

    som.train(
        data=normal_train,
        num_iteration=num_epochs,
        verbose=verbose,
        use_epochs=True,
        random_order=True
    )

    return som

def save_som(som, sidelength, sigma, learning_rate, num_epochs, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #save model
    with open(save_folder + '/model.pkl', 'wb') as file:
        pickle.dump(som, file)

    #save parameters
    parameters = [sidelength, sigma, learning_rate, num_epochs]
    with open(save_folder + '/parameters.pkl', 'wb') as file:
        pickle.dump(parameters, file)



'''
#
bmu_list = [som.winner(x) for x in train_frames]
bmu_counts = Counter(bmu_list)

# Convert counts to a heatmap
hit_map = np.zeros((19, 19))
for (x, y), count in bmu_counts.items():
    hit_map[x, y] = count

plt.figure(figsize=(8, 8))
plt.pcolor(hit_map, cmap='Blues')
plt.colorbar(label="BMU Hits")
plt.title("Best Matching Unit Hit Map")
plt.savefig('bmu-hitmap-hydropower', dpi=300)
'''

if __name__ == '__main__':
    sidelength_h = 19
    sigma_h = 24
    learning_rate_h = 0.5
    num_epochs_h = 5
    #som_hydropower = train_som(sidelength_h, sigma_h, learning_rate_h, num_epochs_h, dataset='Hydropower', verbose=True)
    #save_som(som_hydropower, sidelength_h, sigma_h, learning_rate_h, num_epochs_h, 'Models/Hydropower/base_model')

    sidelength_m = 12
    sigma_m = 15
    learning_rate_m = 0.5
    num_epochs_m = 5
    som_mimii = train_som(sidelength_m, sigma_m, learning_rate_m, num_epochs_m, dataset='Mimii', verbose=True)
    save_som(som_mimii, sidelength_m, sigma_m, learning_rate_m, num_epochs_m, 'Models/MIMII/base_model')






