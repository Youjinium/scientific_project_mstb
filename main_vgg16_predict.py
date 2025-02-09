import os
import tensorflow as tf
import numpy as np
import pandas as pd
from models.vgg_model import load_vgg_model, create_pruned_vgg_model
from utils import plot_confusion_matrix, evaluate_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot

# Paths
DATASET_PATH = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/dataset/test"
WEIGHTS_DIR = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/weights"
RESULTS_PATH = "model_comparison_results.csv"

# Model Paths
UNOPTIMIZED_MODEL_PATH = os.path.join(WEIGHTS_DIR, "vgg16.weights.h5")
PRUNED_MODEL_PATH = os.path.join(WEIGHTS_DIR, "compressed_pruned_vgg16.h5")
QUANTIZED_MODEL_PATH = os.path.join(WEIGHTS_DIR, "quantized_vgg16.tflite")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test dataset
test_ds = image_dataset_from_directory(
    DATASET_PATH,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names  # Retrieve class names from dataset

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
     layers.RandomFlip("vertical"),

])

def measure_model_size(model_path):
    """Returns the model size in MB."""
    return os.path.getsize(model_path) / 1e6  # Convert bytes to MB

# Load models and evaluate
results = []

# Unoptimized Model
unoptimized_model = load_vgg_model(IMG_SIZE, UNOPTIMIZED_MODEL_PATH, data_augmentation)
print("Evaluating Unoptimized model...")
evaluate_model(unoptimized_model, test_ds, class_names)
print("Generating confusion matrix...")
plot_confusion_matrix(unoptimized_model, test_ds, class_names)
results.append(["Unoptimized", measure_model_size(UNOPTIMIZED_MODEL_PATH)])

# Pruned Model
pruned_model, pruning_callback = create_pruned_vgg_model(IMG_SIZE, data_augmentation)
pruned_model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
# Save entire model
pruned_model_stripped.save(PRUNED_MODEL_PATH)  
print("Evaluating Pruned model...")
evaluate_model(pruned_model_stripped, test_ds, class_names)
print("Generating confusion matrix...")
plot_confusion_matrix(pruned_model_stripped, test_ds, class_names)
results.append(["Pruned", measure_model_size(PRUNED_MODEL_PATH)])

# Quantized Model (TFLite)
interpreter = tf.lite.Interpreter(model_path=QUANTIZED_MODEL_PATH)
interpreter.allocate_tensors()
results.append(["Quantized", measure_model_size(QUANTIZED_MODEL_PATH)])

# Save results to CSV
df = pd.DataFrame(results, columns=["Method", "Size (MB)"])
df.to_csv(RESULTS_PATH, index=False)

# Print results in terminal
print("\nComparison Table:")
print(f"{'Method':<20}{'Size (MB)':<15}")
for row in results:
    print(f"{row[0]:<20}{row[1]:<15.2f}")

print("\nResults saved to", RESULTS_PATH)
