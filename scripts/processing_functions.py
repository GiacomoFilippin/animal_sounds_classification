import pandas as pd
import numpy as np
import librosa
import os
import sys
import re
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root

def load_data(dataset_directory=None):
    if dataset_directory is None:
        dataset_directory = os.path.join(project_root, 'data', 'raw')
    X_train_path = os.path.join(dataset_directory, 'train_X.npy')
    y_train_path = os.path.join(dataset_directory, 'train_y.npy')
    X_val_path = os.path.join(dataset_directory, 'val_X.npy')
    y_val_path = os.path.join(dataset_directory, 'val_y.npy')
    X_test_path = os.path.join(dataset_directory, 'test_X.npy')
    y_test_path = os.path.join(dataset_directory, 'test_y.npy')

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    # Reshape the data to fit the model input shape
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)  # (samples, height, width, channels)
    X_val = X_val.reshape(X_val.shape[0], 32, 32, 1)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# creates a dict of sub labels as well, dividing into musical labels and animal labels
# basically it maps the labels to the sub_labels. for example, 0, being a musical instrument, is mapped to "0", while 13, being a frog, is mapped to "1"
def create_labels_dict():
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
    sub_labels_mapping_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 1,
        14: 1,
        15: 1,
        16: 1,
        17: 1,
        18: 1,
        19: 1
    }
    sub_labels_dict = {
        0: "Musical",
        1: "Animal"
    }
    return labels_dict, sub_labels_mapping_dict, sub_labels_dict

def extract_track_id(input_string):
    match = re.search(r'(\d{6})\.mp3$', input_string)
    if match:
        return int(match.group(1))  # Converti a intero per rimuovere gli zeri iniziali
    return None

def load_features(folder=project_root+"\\data\\processed\\fma_metadata"):
    # Carica i metadati
    tracks = pd.read_csv(f"{folder}\\tracks.csv", index_col=0, header=[0, 1])
    genres = pd.read_csv(f"{folder}\\genres.csv", index_col=0)
    features = pd.read_csv(f"{folder}\\features.csv", index_col=0)
    echonest = pd.read_csv(f"{folder}\\echonest.csv", index_col=0)
    
    return tracks, genres, features, echonest

def compute_mel_spectrogram(y, sr = 44100):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    db_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return db_mel_spectrogram

# Calcola lo STFT
def compute_stft_spectrogram(y, n_fft = 2048, hop_length = 512, window = 'hann'):
    D = librosa.stft(y=y, 
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window)
    # Converti in magnitudine (valori assoluti) e quadrato per la potenza
    magnitude = np.abs(D)
    db_stft__spectrogram = librosa.amplitude_to_db(magnitude**2, ref=np.max)
    return db_stft__spectrogram
