from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os 
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
models_directory = os.path.join(project_root, 'models')
sys.path.insert(0, project_root)
from scripts.processing_functions import create_labels_dict
from config import config

def build_model(n_classes=20, n_sub_classes=2, output="classes"):
    """
    Build and compile a CNN model for audio classification using the functional API.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    n_c = {
        "classes": n_classes,
        "sub_classes": n_sub_classes
    }
    # Define the input layer
    inputs = Input(shape=config.model_config['input_shape'])
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
    #shared_dropout = Dropout(0.25)(shared_dense)
    # Output Layer(s)
    if output == "multi":
        # Multi-output model
        classes_output = Dense(n_classes, activation='softmax', name='classes')(shared_dense)
        sub_classes_output = Dense(n_sub_classes, activation='softmax', name='sub_classes')(shared_dense)
        outputs = [classes_output, sub_classes_output]
    else:
        classes_output = Dense(n_c[output], activation='softmax', name=output)(shared_dense)
        outputs = classes_output # Single output
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_metrics_and_losses(output):
    """
    Build metrics and losses for the model.

    Args:
        multi_output (bool): Whether to use multiple outputs.

    Returns:
        metrics (dict): Dictionary of metrics.
        losses (dict): Dictionary of losses.
    """
    if output == "multi":
        metrics = {
            'classes': 'accuracy',
            'sub_classes': 'accuracy'
        }
        losses = {
            'classes': SparseCategoricalCrossentropy(),
            'sub_classes': SparseCategoricalCrossentropy()
        }
        loss_weights = {
            'classes': config.model_config["classes_loss_weight"],
            'sub_classes': config.model_config["sub_classes_loss_weight"]
        }
    else:
        metrics = ['accuracy']
        losses = SparseCategoricalCrossentropy()
        loss_weights = None

    return metrics, losses, loss_weights

class ModelWrapper:
    def __init__(self, output="classes", dataset_directory=None):
        self.output = output
        self.dataset_directory = dataset_directory
        self.lr = config.model_config['learning_rate']
        self.epochs = config.model_config['epochs']
        self.batch_size = config.model_config['batch_size']
        self.early_stopping_patience = config.model_config['early_stopping_patience']

    def data_pipeline(self):
        if self.dataset_directory is None:
            self.dataset_directory = os.path.join(project_root, 'data', 'raw')
        X_train_path = os.path.join(self.dataset_directory, 'train_X.npy')
        y_train_path = os.path.join(self.dataset_directory, 'train_y.npy')
        X_val_path = os.path.join(self.dataset_directory, 'val_X.npy')
        y_val_path = os.path.join(self.dataset_directory, 'val_y.npy')
        X_test_path = os.path.join(self.dataset_directory, 'test_X.npy')
        y_test_path = os.path.join(self.dataset_directory, 'test_y.npy')
        X_train = np.load(X_train_path)
        self.y_train = np.load(y_train_path)
        X_val = np.load(X_val_path)
        self.y_val = np.load(y_val_path)
        X_test = np.load(X_test_path)
        self.y_test = np.load(y_test_path)
        # Reshape the data to fit the model input shape
        self.X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)  # (samples, height, width, channels)
        self.X_val = X_val.reshape(X_val.shape[0], 32, 32, 1)
        self.X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)
        # Load metadata and features.
        self.labels, self.sub_labels_mapping_dict, self.sub_labels = create_labels_dict()
        if self.output == "multi" or self.output == "sub_classes":
            # creates y_train_sub, y_val_sub, y_test_sub using the sub_labels_mapping_dict
            self.y_train_sub = np.array([self.sub_labels_mapping_dict[label] for label in self.y_train])
            self.y_val_sub = np.array([self.sub_labels_mapping_dict[label] for label in self.y_val])
            self.y_test_sub = np.array([self.sub_labels_mapping_dict[label] for label in self.y_test])
    
    def train_model(self):
        self.model = build_model(output=self.output)
        self.metrics, losses, loss_weights = build_metrics_and_losses(output=self.output)
        self.model.compile(optimizer=Adam(learning_rate=self.lr),
                        loss=losses,
                        metrics=self.metrics,
                        loss_weights=loss_weights)
        # Fit the model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Construct the full path for the checkpoint file
        checkpoint_filepath = os.path.join(models_directory, f'model_{timestamp}.h5')
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, restore_best_weights=True)
        # Update ModelCheckpoint to use the new path and filename format
        model_checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss')
        # fit the model
        if self.output == "multi":
            self.y_train_sub = np.array([self.sub_labels_mapping_dict[label] for label in self.y_train])
            self.y_val_sub = np.array([self.sub_labels_mapping_dict[label] for label in self.y_val])
            self.y_test_sub = np.array([self.sub_labels_mapping_dict[label] for label in self.y_test])
            # Create a dictionary for multi-output
            self.y_t = {
                'classes': self.y_train,
                'sub_classes': self.y_train_sub
            }
            self.y_v = {
                'classes': self.y_val,
                'sub_classes': self.y_val_sub
            }
        elif self.output == "classes":
            self.y_t = self.y_train
            self.y_v = self.y_val
        elif self.output == "sub_classes":
            self.y_t = self.y_train_sub
            self.y_v = self.y_val_sub
        else:
            raise ValueError("Invalid output type. Choose 'classes', 'sub_classes', or 'multi'.")
        self.history = self.model.fit(self.X_train, self.y_t, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.X_val, self.y_v), callbacks=[early_stopping, model_checkpoint])

    def plot_training_history(self):
        if self.output == "multi":
            metrics = ['classes_accuracy', 'val_classes_accuracy',
                       'sub_classes_accuracy', 'val_sub_classes_accuracy',]
            losses = ['classes_loss', 'val_classes_loss', 
                      'sub_classes_loss', 'val_sub_classes_loss']
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            # Plot classes accuracy
            plt.plot(self.history.history[metrics[0]], label='Train Classes Accuracy')
            plt.plot(self.history.history[metrics[1]], label='Validation Classes Accuracy')
            # Plot sub_classes accuracy
            plt.plot(self.history.history[metrics[2]], label='Train Sub-Classes Accuracy', linestyle='--')
            plt.plot(self.history.history[metrics[3]], label='Validation Sub-Classes Accuracy', linestyle='--')
            plt.title('Model Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(1, 2, 2)
            # Plot classes loss
            plt.plot(self.history.history[losses[0]], label='Train Classes Loss')
            plt.plot(self.history.history[losses[1]], label='Validation Classes Loss')
            # Plot sub_classes loss
            plt.plot(self.history.history[losses[2]], label='Train Sub-Classes Loss', linestyle='--')
            plt.plot(self.history.history[losses[3]], label='Validation Sub-Classes Loss', linestyle='--')
            plt.title('Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
        else:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
        return plt
    
    def evaluate_model(self):
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
        history_plot = self.plot_training_history()
        history_plot.show()
        if self.output == "multi":
            # Evaluate the model on the test set for multi-output
            y_test_dict = {'classes': self.y_test, 'sub_classes': self.y_test_sub}
            results = self.model.evaluate(self.X_test, y_test_dict, verbose=0)
            # The order of results is [total_loss, classes_loss, sub_classes_loss, classes_accuracy, sub_classes_accuracy]
            # based on the order in model.compile(loss=..., metrics=...)
            total_loss = results[0]
            classes_loss = results[1]
            sub_classes_loss = results[2]
            classes_accuracy = results[3]
            sub_classes_accuracy = results[4]

            print("--- Multi-Output Evaluation ---")
            print(f"Total Test Loss: {total_loss:.4f}")
            print(f"Classes Test Loss: {classes_loss:.4f}")
            print(f"Sub-Classes Test Loss: {sub_classes_loss:.4f}")
            print(f"Classes Test Accuracy: {classes_accuracy:.4f}")
            print(f"Sub-Classes Test Accuracy: {sub_classes_accuracy:.4f}")

            # Get predictions (returns a list of arrays, one for each output)
            y_pred_list = self.model.predict(self.X_test)
            y_pred_classes = np.argmax(y_pred_list[0], axis=1) # Predictions for 'classes'
            y_pred_sub_classes = np.argmax(y_pred_list[1], axis=1) # Predictions for 'sub_classes'

            # Confusion Matrix for 'classes'
            cm_classes = confusion_matrix(self.y_test, y_pred_classes)
            cm_df_classes = pd.DataFrame(cm_classes, index=self.labels.values(), columns=self.labels.values())
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm_df_classes, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Classes')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

            # Confusion Matrix for 'sub_classes'
            cm_sub_classes = confusion_matrix(self.y_test_sub, y_pred_sub_classes)
            cm_df_sub_classes = pd.DataFrame(cm_sub_classes, index=self.sub_labels.values(), columns=self.sub_labels.values())
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm_df_sub_classes, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Sub-Classes')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
        elif self.output == "classes":
            # Evaluate the model on the test set for single 'classes' output
            test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            print(f"--- Classes Evaluation ---")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")

            # print the confusion matrix
            y_pred = self.model.predict(self.X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            cm = confusion_matrix(self.y_test, y_pred_classes)
            cm_df = pd.DataFrame(cm, index=self.labels.values(), columns=self.labels.values())
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Classes')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

        elif self.output == "sub_classes":
            # Evaluate the model on the test set for single 'sub_classes' output
            test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test_sub, verbose=0)
            print(f"--- Sub-Classes Evaluation ---")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")

            # print the confusion matrix
            y_pred = self.model.predict(self.X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            cm = confusion_matrix(self.y_test_sub, y_pred_classes)
            cm_df = pd.DataFrame(cm, index=self.sub_labels.values(), columns=self.sub_labels.values())
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Sub-Classes')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

    def run(self):
        print("Loading data...")
        self.data_pipeline()
        print("Data loaded.")
        print("Starting training...")
        self.train_model()
        print("Training completed.")
        print("Evaluating model...")
        self.evaluate_model()

if __name__ == "__main__":
    model = ModelWrapper(output="multi")
    model.run()