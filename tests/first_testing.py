#%%

import os 
import sys
import numpy as np
# Aggiungi la root del progetto al Python path
project_root = os.path.dirname(os.path.abspath(''))
models_directory = os.path.join(project_root, 'models')
sys.path.append(project_root)  # Ora Python trova i moduli nella root
from scripts.plotting_functions import make_mel_spectrogram_plot, plot_mel_spectrogram, plot_given_labels_spectrograms, plot_training_history
from scripts.processing_functions import create_labels_dict, extract_track_id, load_data
from main_code.models import train_model, evaluate_model
# %% Load metadata and features.
X_train, y_train, X_val, y_val, X_test, y_test = load_data()
labels, sub_labels_mapping_dict, sub_labels = create_labels_dict()
# creates y_train_sub, y_val_sub, y_test_sub using the sub_labels_mapping_dict
# %%
plot_given_labels_spectrograms(X_test, y_test, labels_dict=labels, requested_label="Guitar_3rd_Fret")
# %% define and fit the model
history, model = train_model(X_train, y_train, X_val, y_val, n_classes=20)
# %% plot the training history
evaluate_model(model, history, X_test, y_test, labels, sub_labels, output="clases")