import pandas as pd
import numpy as np
import librosa
import os
import sys
import re
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root

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
