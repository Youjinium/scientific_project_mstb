import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from models.mobilenet_model import create_mobilenet_model, create_pruned_mobilenet_model
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
UNOPTIMIZED_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "mobilenet.weights.h5")
COMPRESSED_PRUNED_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "compressed_pruned_mobilenetv2.h5")
QUANTIZED_MODEL_PATH = os.path.join(WEIGHTS_DIR, "quantized_mobilenetv2.tflite")
UNOPTIMIZED_TRAINING_RESULT_CSV_PATH = os.path.join(HISTORY_DIR, "mobilenet_training_history.csv")

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

class_names = test_ds.class_names

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2)
])

def save_compressed_pruned_model(pruned_model, save_path):
    compressed_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    compressed_model.save(save_path)
    print(f"Compressed pruned model saved to {save_path}")
    return os.path.getsize(save_path) / 1e6

def convert_to_tflite_quantized_model(weights_path, quantized_model_path):
    print("Loading model for quantization...")
    model = create_mobilenet_model(IMG_SIZE, data_augmentation)
    model.load_weights(weights_path)

    print("Converting model to TensorFlow Lite with quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(quantized_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Quantized model saved to {quantized_model_path}")

def train_and_compare_mobilenet():
    print("Training Unoptimized MobileNetV2 model...")
    unoptimized_model = train_model("mobilenet")
    
    unoptimized_model_accuracy, unoptimized_model_loss = extract_final_accuracy_loss_from_csv(UNOPTIMIZED_TRAINING_RESULT_CSV_PATH)
    unoptimized_model_size = os.path.getsize(UNOPTIMIZED_WEIGHTS_PATH) / 1e6  

    print("Training Pruned MobileNetV2 model...")
    pruned_model, pruning_callback = create_pruned_mobilenet_model(IMG_SIZE, data_augmentation)
    pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history_pruned = pruned_model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[pruning_callback, EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)]
    )
    
    pruned_model_size = save_compressed_pruned_model(pruned_model, COMPRESSED_PRUNED_WEIGHTS_PATH)
    save_training_history(history_pruned, os.path.join(HISTORY_DIR, "pruned_mobilenetv2_training_history.csv"))
    
    plot_accuracy(history_pruned)

    print("Evaluating Pruned MobileNetV2 model...")
    evaluate_model(pruned_model, test_ds, class_names)
    plot_confusion_matrix(pruned_model, test_ds, class_names)
    
    convert_to_tflite_quantized_model(COMPRESSED_PRUNED_WEIGHTS_PATH, QUANTIZED_MODEL_PATH)
    quantized_model_size = os.path.getsize(QUANTIZED_MODEL_PATH) / 1e6  
    
    quantized_accuracy = evaluate_tflite_model(QUANTIZED_MODEL_PATH, test_ds, class_names)

    print("\nComparison Table:")
    print(f"{'Method':<20}{'Size (MB)':<15}{'Accuracy':<15}{'Loss':<15}")
    print(f"{'Unoptimized':<20}{unoptimized_model_size:<15.2f}{unoptimized_model_accuracy:<15.2f}{unoptimized_model_loss:<15.2f}")
    print(f"{'Pruned (Compressed)':<20}{pruned_model_size:<15.2f}{history_pruned.history['accuracy'][-1]:<15.2f}{history_pruned.history['loss'][-1]:<15.2f}")
    print(f"{'Pruned + Compressed + Quantized (TFLite)':<20}{quantized_model_size:<15.2f}{quantized_accuracy:<15.2f}{'N/A':<15}")

if __name__ == "__main__":
    print("Running MobileNetV2 Model Training & Comparison with Quantization...")
    train_and_compare_mobilenet()