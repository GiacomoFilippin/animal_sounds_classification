model_config = {
    "input_shape": (32, 32, 1),  # Shape of the input data (height, width, channels)
    "batch_size": 64,  # Batch size for training
    "epochs": 50,  # Number of epochs for training
    "learning_rate": 0.0001,  # Learning rate for the optimizer
    "early_stopping_patience": 5,  # Patience for early stopping
    "classes_loss_weight": 0.75,  # Loss weight for the main classes
    "sub_classes_loss_weight": 1.0,  # Loss weight for the sub-classes
}