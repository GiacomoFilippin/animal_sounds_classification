# in this script, we will test and explore the feature space, 
# with the goal of deciding which features are suited for our multitask study 
# %%
import zipfile
import io
import os 
import sys
import numpy as np
# Aggiungi la root del progetto al Python path
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root
from scripts.processing_functions import load_features
# %% Load metadata and features.
tracks_data, genres_data, features_data, echonest_data = load_features()
print('Echonest features available for {} tracks.'.format(len(echonest_data)))
# %% pair features and tracks
features_clean = features_data.iloc[3:]
# Converti gli indici di features_clean in interi
features_clean.index = features_clean.index.astype(int)
# Ora il test dovrebbe passare
np.testing.assert_array_equal(features_clean.index, tracks_data.index)
tracks_medium = tracks_data[tracks_data['set', 'subset'] <= 'medium']
print(tracks_medium.columns)
# %% explore genres
genres_top = tracks_medium['track', 'genre_top']
genres = tracks_medium['track', 'genres']
genres_all = tracks_medium['track', 'genres_all']
genres_top.shape
# try to get an idea if they can be used.
# it don't matter how well the classification works,
# it would just matter if it improves when adding "intermediate outputs"
# %%
tracks_medium.shape, features_clean.shape, echonest_data.shape

durations = tracks_medium['track', 'duration']
genres = tracks_medium['track', 'genre_top']
# %%
tracks_medium.index