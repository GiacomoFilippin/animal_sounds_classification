import matplotlib.pyplot as plt
import librosa.display

def make_stft_spectrogram_plot(spectrogram, sr=44100, title="STFT Power Spectrogram", hop_length=512, n_fft=2048, fmax=8192):
    plt.figure(figsize=(12, 6))
    # Crea il plot dello spettrogramma
    librosa.display.specshow(spectrogram, 
                            sr=sr,
                            x_axis='time',
                            y_axis="linear",
                            hop_length=hop_length,
                            n_fft=n_fft,
                            fmax=sr//2,
                            cmap='viridis')

    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Power Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 5000)  # Regola questo in base all'interesse frequenziale
    plt.tight_layout()
    return plt

def make_mel_spectrogram_plot(spectrogram, sr=44100, title="Mel Power Spectrogram", fmax=8192):
    plt.figure(figsize=(12, 6))
    # Crea il plot dello spettrogramma
    librosa.display.specshow(spectrogram, 
                            sr=sr,
                            x_axis='time',
                            y_axis='mel',
                            fmax=fmax,
                            cmap='viridis')

    # Aggiungi barra del colore e labels
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    
    return plt

def plot_mel_spectrogram(spectrogram, sr=44100, title="Mel Spectrogram (32x32)", hop_length=344, n_fft=1024, fmax=8192):
    """
    Plot a precomputed 32x32 Mel spectrogram.
    
    Args:
        spectrogram (np.ndarray): 32x32 Mel spectrogram (frequency × time)
        sr (int): Sample rate (default: 44100)
        hop_length (int): Samples between frames (default: 344 for 0.25s/32 bins)
        n_fft (int): FFT window size (default: 1024)
        fmax (int): Max frequency to display (default: 8192)
    """
    plt.figure(figsize=(8, 4))
    
    # Display with explicit time/frequency axes
    librosa.display.specshow(
        spectrogram,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=n_fft,
        x_axis='time',
        y_axis='mel',
        fmax=fmax,
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()

    return plt


def plot_given_labels_spectrograms(X, y, labels_dict, requested_label="Bufo_Alvarius"):
    for i, label in enumerate(y):
        if labels_dict[label] == requested_label:
            plt = plot_mel_spectrogram(X[i, :, :].T, title=labels_dict[label])
            plt.show()
            plt.close()

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    return plt