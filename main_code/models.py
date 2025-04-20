from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os 
import datetime
import numpy as np
from config.config import model_config
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
project_root = os.path.dirname(os.path.abspath(''))
models_directory = os.path.join(project_root, 'models')
sys.path.append(project_root)  # Ora Python trova i moduli nella root
from scripts.plotting_functions import plot_training_history

def build_model(n_classes=20, n_sub_classes=2, multi_output=False):
    """
    Build and compile a CNN model for audio classification using the functional API.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    # Define the input layer
    inputs = Input(shape=model_config['input_shape'])
    # Convolutional Block 1
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    # Convolutional Block 2
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    # Shared Flatten and Dense Layers
    x = Flatten()(x)
    shared_dense = Dense(128, activation='relu')(x)
    shared_dropout = Dropout(0.5)(shared_dense)
    # Output Layer(s)
    if multi_output:
        classes_output = Dense(n_classes, activation='softmax', name='classes')(shared_dropout)
        sub_classes_output = Dense(n_sub_classes, activation='softmax', name='sub_classes')(shared_dropout)
        outputs = [classes_output, sub_classes_output]
    else:
        classes_output = Dense(n_classes, activation='softmax', name='classes')(shared_dropout)
        outputs = classes_output # Single output
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(X_train, y_train, X_val, y_val, n_classes):
    model = build_model(n_classes)
    model.compile(optimizer=Adam(learning_rate=model_config['learning_rate']),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Construct the full path for the checkpoint file
    checkpoint_filepath = os.path.join(models_directory, f'model_{timestamp}.h5')
    early_stopping = EarlyStopping(monitor='val_loss', patience=model_config["early_stopping_patience"], restore_best_weights=True)
    # Update ModelCheckpoint to use the new path and filename format
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss')
    # fit the model
    history = model.fit(X_train, y_train, epochs=model_config["epochs"], batch_size=model_config["batch_size"], validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

    return history, model

def evaluate_model(model, history, X_test, y_test, labels):
    """
    Evaluate the model on the test set.

    Args:
        model (tf.keras.Model): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        test_loss (float): Loss on the test set.
        test_accuracy (float): Accuracy on the test set.
    """
    history_plot = plot_training_history(history)
    history_plot.show()
    # evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    # print the confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    cm_df = pd.DataFrame(cm, index=labels.values(), columns=labels.values())
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    return test_loss, test_accuracy