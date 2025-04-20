# in this script, we will test and explore the zipped audio dataset,
# with the goal of creating our spectrograms without decompressing everything.
# possibly, a pre filtering having our target dataset with valid tracks' ids would be very nice to save some extra space

# on hold untill i gathered the feature dataset!
#%% explore mp3s zip
import os
import zipfile
import librosa
import io
import pickle
import sys
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root
from scripts.processing_functions import compute_mel_spectrogram, compute_stft_spectrogram, extract_track_id
from scripts.plotting_functions import make_stft_spectrogram_plot, make_mel_spectrogram_plot
# %%processed  track_spectrogram.pkl
dataset_directory = os.path.join('..', 'data', 'processed', 'track_spectrogram.pkl')
# Crea la directory di output se non esiste
os.makedirs(os.path.dirname(dataset_directory), exist_ok=True)
# Inizializza il dizionario
spectrogram_dataset = {}
zip_path = os.path.join('..', 'data', 'raw', 'fma_medium.zip')
track_numbers = []
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List all files in the archive
    print(zip_ref.namelist())
    track_list = [name for name in zip_ref.namelist() if "mp3" in name]
    print(track_list)
    with zip_ref.open("fma_medium/README.txt") as file_in_zip:
        audio_bytes = file_in_zip.read()
    for i, track in enumerate(track_list):
        try:
            # Read a specific file (e.g., 'data/track1.mp3') as bytes
            with zip_ref.open(track) as file_in_zip:
                audio_bytes = file_in_zip.read()
            track_n = extract_track_id(track)
            track_numbers.append(track_n)
            # insert checking to filter tracks with no features
            print(f"processing track {track_n}. \ntrack {i} of {len(track_list)}")
            # Process the file (e.g., convert to spectrogram)
            """y, sr = librosa.load(io.BytesIO(audio_bytes), sr=44100)
            # insert bass_band filtering
            stft_spectrogram = compute_stft_spectrogram(y=y)
            stft_spectrogram_plot = make_stft_spectrogram_plot(spectrogram=stft_spectrogram, sr=sr)
            stft_spectrogram_plot.show()
            # Aggiungi al dataset
            spectrogram_dataset[os.path.basename(track)] = stft_spectrogram  # Usa solo il nome del file come chiave
                """
        except Exception as e:
            print(f"Error processing {track}: {str(e)}")
            continue
print(f"track ids found: {track_numbers}")
"""# Salva il dataset
with open(dataset_directory, 'wb') as f:
    pickle.dump(spectrogram_dataset, f)

print(f"Dataset salvato correttamente in {dataset_directory}")"""

# %%
