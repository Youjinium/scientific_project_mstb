import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow_model_optimization as tfmot




# Function to create a model
def create_model(base_model, preprocess_input, img_size, data_augmentation, num_classes=3, dropout_rate=0.3):
    """
    General function to create a transfer learning model.

    Parameters:
    - base_model: Pre-trained model to use as the backbone.
    - preprocess_input: Preprocessing function for the chosen model.
    - img_size: Tuple, image dimensions (height, width).
    - data_augmentation: A `tf.keras.Sequential` object for data augmentation.
    - num_classes: Number of output classes (default is 3).
    - dropout_rate: Dropout rate for regularization (default is 0.3).

    Returns:
    - A compiled `tf.keras.Model` ready for training.
    """
    base_model.trainable = False
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model


# Compile a model
def compile_model(model, learning_rate=1e-4):
    """
    Compiles a given model with standard parameters.

    Parameters:
    - model: The model to compile.
    - learning_rate: Learning rate for the optimizer (default is 1e-4).

    Returns:
    - None (model is compiled in-place).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


# Save training history to a CSV
def save_training_history(history, filename):
    """
    Save the training history to a CSV file.

    Parameters:
    history: History object returned by model.fit().
    filename: Name of the file where the history will be saved.
    """
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(filename, index=False)
    print(f"Training history saved to {filename}")


# Load training history from a CSV
def load_training_history(filename):
    """
    Read the training history from a CSV file.

    Parameters:
    filename: Name of the file to read the history from.

    Returns:
    DataFrame containing the training history.
    """
    history_df = pd.read_csv(filename)
    return history_df


# Plot training and validation accuracy and loss from a history object
def plot_accuracy(history):
    """
    Plot training and validation accuracy and loss from a history object.

    Parameters:
    history: History object returned by model.fit().
    """
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Plot accuracy and loss from a CSV
def plot_accuracy_from_csv(history_df):
    """
    Plot training and validation accuracy and loss from a DataFrame.

    Parameters:
    history_df: DataFrame containing the training history (e.g., from a CSV file).
    """
    plt.figure(figsize=(10, 6))

    # Accuracy
    plt.plot(history_df['accuracy'], label='Train Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')

    # Loss
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')

    plt.title("Model Performance")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# Evaluate model on a test dataset
def evaluate_model(model, test_ds, class_names):
    """
    Evaluate a model on a test dataset and print the results.

    Parameters:
    model: The model to evaluate.
    test_ds: The test dataset.
    class_names: List of class names.

    Returns:
    - Tuple (correct_predictions, failed_predictions, total_samples, accuracy)
    """
    total_samples = 0
    correct_predictions = 0
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        correct_predictions += np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
        total_samples += len(labels)

    accuracy = correct_predictions / total_samples
    failed_predictions = total_samples - correct_predictions

    print(f"Correct Predictions: {correct_predictions}")
    print(f"Failed Predictions: {failed_predictions}")
    print(f"Total Test Samples: {total_samples}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return correct_predictions, failed_predictions, total_samples, accuracy


# Plot confusion matrix
def plot_confusion_matrix(model, test_ds, class_names=None):
    """
    Plot the confusion matrix for a model's predictions on a test dataset.

    Parameters:
    model: The trained model.
    test_ds: The test dataset.
    class_names: List of class names (optional).
    """
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()




def save_compressed_pruned_model(pruned_model, save_path):
    """
    Strips pruning wrappers and saves the compressed pruned model.
    """
    compressed_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    compressed_model.save(save_path)
    print(f"Compressed pruned model saved to {save_path}")
    return os.path.getsize(save_path) / 1e6  # Convert to MB

def convert_to_tflite_quantized_model(weights_path, quantized_model_path, create_vgg_model, img_size, data_augmentation):
    """
    Converts a model to TensorFlow Lite format with quantization.
    """
    print("Loading model for quantization...")
    model = create_vgg_model(img_size, data_augmentation)
    model.load_weights(weights_path)
    print(f"Loaded weights from {weights_path}")

    # Convert model to TensorFlow Lite with quantization
    print("Converting model to TensorFlow Lite with quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  
    tflite_model = converter.convert()

    with open(quantized_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Quantized model saved to {quantized_model_path}")


def compare_model_sizes(model_paths):
    """
    Compares the size of different models.
    """
    for name, path in model_paths.items():
        size_mb = os.path.getsize(path) / 1e6
        print(f"{name}: {size_mb:.2f} MB")




def plot_confusion_matrix_from_preds(y_true, y_pred, class_names):
    """
    Plots the confusion matrix given true and predicted labels.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - class_names: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (TFLite Model)')
    plt.show()


def evaluate_tflite_model(tflite_model_path, test_ds, class_names):
    """
    Evaluates the TensorFlow Lite quantized model manually.

    Parameters:
    - tflite_model_path: Path to the quantized model.
    - test_ds: The test dataset.
    - class_names: List of class names.

    Returns:
    - Tuple (correct_predictions, failed_predictions, total_samples, accuracy)
    """
    print(f"Evaluating quantized model from {tflite_model_path}...")

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    total_samples = 0
    correct_predictions = 0
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        images = images.numpy()
        labels = np.argmax(labels.numpy(), axis=1)

        for i in range(len(images)):
            input_data = np.expand_dims(images[i], axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(output_data)

            y_pred.append(predicted_label)
            y_true.append(labels[i])

            if predicted_label == labels[i]:
                correct_predictions += 1
            total_samples += 1

    accuracy = correct_predictions / total_samples
    failed_predictions = total_samples - correct_predictions

    # Print results (same as `evaluate_model()`)
    print(f"\nEvaluation Results for Quantized Model:")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Failed Predictions: {failed_predictions}")
    print(f"Total Test Samples: {total_samples}")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    # Plot confusion matrix
    plot_confusion_matrix_from_preds(y_true, y_pred, class_names)

    return accuracy

def extract_final_accuracy_loss_from_csv(file_path):
    """
    Extracts the final accuracy and loss from a training history CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the training history.

    Returns:
    tuple: Final accuracy and loss values.
    """
    # Load the CSV file
    history_df = pd.read_csv(file_path)
    
    # Extract last row values
    final_accuracy = history_df.iloc[-1]["accuracy"]
    final_loss = history_df.iloc[-1]["loss"]
    
    return final_accuracy, final_loss





