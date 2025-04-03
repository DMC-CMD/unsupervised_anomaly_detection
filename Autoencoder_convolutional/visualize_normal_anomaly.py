import librosa
import numpy as np
from matplotlib import pyplot as plt

test_spectrograms = np.load('../Spectrograms/Hydropower/normal_test_spectrograms.npy')
normal_example = test_spectrograms[12]
normal_example = np.squeeze(normal_example)

anomaly_spectrograms_clapping = np.load('../../Spectrograms/Hydropower/anomaly_spectrograms_clapping.npy')
anomaly_example = anomaly_spectrograms_clapping[12]
anomaly_example = np.squeeze(anomaly_example)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

img = librosa.display.specshow(normal_example, x_axis='time', y_axis='mel', sr=22050, hop_length=256, n_fft=1024, ax=ax[0])
ax[0].set(title='Normal Spectrogram')
ax[0].label_outer()

librosa.display.specshow(anomaly_example, x_axis='time', y_axis='mel', sr=22050, hop_length=256, n_fft=1024, ax=ax[1])
ax[1].set(title='Anomaly Spectrogram')
ax[1].label_outer()
fig.colorbar(img, ax=ax)
plt.show()
