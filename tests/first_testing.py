#%%
import zipfile
import io
import os 
import sys
import numpy as np
# Aggiungi la root del progetto al Python path
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root
from scripts.plotting_functions import make_mel_spectrogram_plot, plot_mel_spectrogram
# %% Load metadata and features.
dataset_directory = os.path.join('..', 'data', 'raw')
X_train_path = os.path.join(dataset_directory, 'train_X.npy')
y_train_path = os.path.join(dataset_directory, 'train_y.npy')
X_val_path = os.path.join(dataset_directory, 'val_X')
y_val_path = os.path.join(dataset_directory, 'val_y')
X_test_path = os.path.join(dataset_directory, 'test_X.npy')
y_test_path = os.path.join(dataset_directory, 'test_y.npy')

X_train = np.load(X_train_path)
y_train = np.load(y_train_path)
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

labels_dict = {
    0: "Drum_FloorTom",
    1: "Drum_HiHat",
    2: "Drum_Kick",
    3: "Drum_MidTom",
    4: "Drum_Ride",
    5: "Drum_Rim",
    6: "Drum_SmallTom",
    7: "Drum_Snare",
    8: "Guitar_3rd_Fret",
    9: "Guitar_9th_Fret",
    10: "Guitar_Chord1",
    11: "Guitar_Chord2",
    12: "Guitar_7th_Fret",
    13: "Bufo_Alvarius",
    14: "Bufo_Canorus",
    15: "Pseudacris_Crucifer",
    16: "Allonemobius_Allardi",
    17: "Anaxipha_Exigua",
    18: "Amblycorypha_Carinata",
    19: "Belocephalus_Sabalis"
}
# %%
def plot_given_labels_spectrograms(X, y, labels_dict, requested_label="Bufo_Alvarius"):
    for i, label in enumerate(y):
        if labels_dict[label] == requested_label:
            plt = plot_mel_spectrogram(X[i, :, :], title=labels_dict[label])
            plt.show()
            plt.close()

plot_given_labels_spectrograms(X_test, y_test, labels_dict=labels_dict, requested_label="Guitar_3rd_Fret")
# %% Crea la directory di output se non esiste
np.shape(X_train)
for i in range(np.shape(X_test)[0]): 
    plt = plot_mel_spectrogram(X_test[i, :, :])
    plt.show()
# %% feature processing

# %%
# Carica i dati
spettrogrammi = np.load(X_train_path)

# Stampa statistiche
print("Min:", np.min(spettrogrammi), "Max:", np.max(spettrogrammi), "Mean:", np.mean(spettrogrammi))

# Conta gli zeri
percentuale_zeri = 100 * np.sum(spettrogrammi == 0) / spettrogrammi.size
print(f"Percentuale di zeri: {percentuale_zeri:.2f}%")
# %%
