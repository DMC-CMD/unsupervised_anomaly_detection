import librosa
import numpy as np
from matplotlib import pyplot as plt
from constants import HOP_LENGTH, SAMPLE_RATE, FRAME_SIZE

normal_spectrograms = np.load('../Spectrograms/Hydropower/normal_test_spectrograms.npy')
normal_example = normal_spectrograms[12]
normal_example = np.squeeze(normal_example)

anomaly_spectrograms = np.load('../Spectrograms/Hydropower/anomaly_test_spectrograms.npy')
anomaly_example = anomaly_spectrograms[10]
anomaly_example = np.squeeze(anomaly_example)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

img = librosa.display.specshow(normal_example, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=FRAME_SIZE, ax=ax[0])
ax[0].set(title='Normal Spectrogram')
ax[0].label_outer()

librosa.display.specshow(anomaly_example, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=FRAME_SIZE, ax=ax[1])
ax[1].set(title='Anomaly Spectrogram')
ax[1].label_outer()
fig.colorbar(img, ax=ax)
plt.show()
