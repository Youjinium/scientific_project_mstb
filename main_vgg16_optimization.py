import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from models.vgg_model import create_vgg_model, create_pruned_vgg_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from main import train_model
from utils import save_training_history, compile_model, plot_accuracy, plot_confusion_matrix, evaluate_model, evaluate_tflite_model, extract_final_accuracy_loss_from_csv

# Paths
DIR_TRAIN = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/dataset/train"
DIR_VAL = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/dataset/val"
DIR_TEST = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/dataset/test"
WEIGHTS_DIR = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/weights"
HISTORY_DIR = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/training_histories"

# Ensure directories exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Model Paths
UNOPTIMIZED_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "vgg16.weights.h5")
COMPRESSED_PRUNED_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "compressed_pruned_vgg16.h5")
QUANTIZED_MODEL_PATH = os.path.join(WEIGHTS_DIR, "quantized_vgg16.tflite")
UNOPTIMIZED_VGG_TRAINING_RESULT_CSV_PATH = os.path.join(HISTORY_DIR, "vgg16_training_history.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Load Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DIR_TRAIN, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DIR_VAL, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DIR_TEST, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

class_names = test_ds.class_names  # Retrieve class names from dataset

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomFlip("vertical")
])

# Early stopping & learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

def save_compressed_pruned_model(pruned_model, save_path):
    """
    Strips pruning wrappers and saves the compressed pruned model.
    """
    compressed_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    compressed_model.save(save_path)
    print(f"Compressed pruned model saved to {save_path}")
    return os.path.getsize(save_path) / 1e6  # Size in MB

def convert_to_tflite_quantized_model(weights_path, quantized_model_path):
    """
    Converts a model to TensorFlow Lite format with quantization.
    """
    print("Loading model for quantization...")
    model = create_vgg_model(IMG_SIZE, data_augmentation)
    model.load_weights(weights_path)

    # Convert model to TensorFlow Lite with quantization
    print("Converting model to TensorFlow Lite with quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
    tflite_model = converter.convert()

    # Save the quantized model
    with open(quantized_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Quantized model saved to {quantized_model_path}")

def train_pruned_model():
    """
    Train the pruned version of the model and save the weights, history, and evaluation results.
    """
    model_name = "pruned_vgg16"
    
    # Create pruned model
    pruned_model, pruning_callback = create_pruned_vgg_model(IMG_SIZE, data_augmentation)
    
    # Compile model
    compile_model(pruned_model)
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    
    # Train model
    print(f"Training {model_name.capitalize()} model...")
    history = pruned_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr, pruning_callback]
    )
    HISTORY_DIR = "training_histories"
    save_training_history(history, os.path.join(HISTORY_DIR, f"{model_name}_training_history.csv"))
    
    # Save compressed pruned weights
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{model_name}.weights.h5")
    compressed_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    compressed_model.save_weights(weights_path)
    print(f"Pruned weights saved to {weights_path}")
    
    # Plot training and validation accuracy/loss
    plot_accuracy(history)
    
    # Evaluate model
    print(f"Evaluating {model_name.capitalize()} model...")
    class_names = test_ds.class_names
    evaluate_model(compressed_model, test_ds, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(compressed_model, test_ds, class_names)
    print(f"{model_name.capitalize()} model evaluation and plots completed.")


def train_and_compare_vgg():
    """
    Trains the unoptimized and pruned VGG16 models and converts to a quantized model.
    """

    # Train Unoptimized Model
    print("Training Unoptimized VGG16 model...")
    unoptimized_model = train_model("vgg16")
    
    unoptimized_model_accuracy, unoptimized_model_loss = extract_final_accuracy_loss_from_csv(UNOPTIMIZED_VGG_TRAINING_RESULT_CSV_PATH)
    unoptimized_model_size = os.path.getsize(UNOPTIMIZED_WEIGHTS_PATH) / 1e6  

    # Train Pruned Model
    print("Training Pruned VGG16 model...")
    pruned_model, pruning_callback = create_pruned_vgg_model(IMG_SIZE, data_augmentation)
    pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history_pruned = pruned_model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[pruning_callback, early_stopping, reduce_lr]
    )
    
    # Save Weights & Training History
    pruned_model_size = save_compressed_pruned_model(pruned_model, COMPRESSED_PRUNED_WEIGHTS_PATH)
    save_training_history(history_pruned, os.path.join(HISTORY_DIR, "pruned_vgg16_training_history.csv"))
    
    # Plot Training Performance
    plot_accuracy(history_pruned)

    # Evaluate Pruned Model
    print("Evaluating Pruned VGG16 model...")
    evaluate_model(pruned_model, test_ds, class_names)
    plot_confusion_matrix(pruned_model, test_ds, class_names)

    # Convert to Quantized Model
    convert_to_tflite_quantized_model(COMPRESSED_PRUNED_WEIGHTS_PATH, QUANTIZED_MODEL_PATH)
    quantized_model_size = os.path.getsize(QUANTIZED_MODEL_PATH) / 1e6  

    # Evaluate Quantized Model
    quantized_accuracy = evaluate_tflite_model(QUANTIZED_MODEL_PATH, test_ds, class_names)

    # Comparison Table
    print("\nComparison Table:")
    print(f"{'Method':<20}{'Size (MB)':<15}{'Accuracy':<15}{'Loss':<15}")
    print(f"{'Unoptimized':<20}{unoptimized_model_size:<15.2f}{unoptimized_model_accuracy:<15.2f}{unoptimized_model_loss:<15.2f}")
    print(f"{'Pruned (Compressed)':<20}{pruned_model_size:<15.2f}{history_pruned.history['accuracy'][-1]:<15.2f}{history_pruned.history['loss'][-1]:<15.2f}")
    print(f"{'Pruned + Compressed + Quantized (TFLite)':<20}{quantized_model_size:<15.2f}{quantized_accuracy:<15.2f}{'N/A':<15}")

if __name__ == "__main__":
    print("Running VGG16 Model Training & Comparison with Quantization...")
    train_and_compare_vgg()

