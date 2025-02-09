import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from utils import evaluate_model, plot_confusion_matrix, load_training_history, plot_accuracy_from_csv
from models.efficientnet_model import load_efficientnet_model
from models.mobilenet_model import load_mobilenet_model
from models.resnet_model import load_resnet_model
from models.vgg_model import load_vgg_model

# Paths
DIR_TEST = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/dataset/test"
WEIGHTS_DIR = "weights"
HISTORY_DIR = "training_histories"

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test dataset
test_ds = image_dataset_from_directory(
    DIR_TEST,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names  # Retrieve class names from dataset

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical")
])


def get_flops(model):
    """
    Compute the FLOPs of a given TensorFlow/Keras model.
    
    Parameters:
    - model: The TensorFlow model for which to compute FLOPs.
    
    Returns:
    - flops: The total number of floating point operations (GFLOPs).
    """
    # Convert model to a Concrete Function
    concrete_func = tf.function(model).get_concrete_function(
        tf.TensorSpec([1] + list(IMG_SIZE) + [3], dtype=tf.float32)
    )

    # Create TensorFlow Graph
    frozen_func = concrete_func.graph

    # Compute FLOPs using TensorFlow Profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func, run_meta=run_meta, options=opts
    )

    total_flops = flops.total_float_ops if flops is not None else 0
    return total_flops / 1e9  # Convert to GFLOPs


def load_model(model_name):
    """
    Load the specified model along with its training history path.

    Parameters:
    - model_name: Name of the model to load (e.g., 'efficientnet', 'mobilenet', 'resnet', 'vgg16').

    Returns:
    - model: The loaded model with weights.
    - history_path: The path to the training history CSV file for the model.
    """
    if model_name == "efficientnet":
        model = load_efficientnet_model(
            IMG_SIZE, 
            os.path.join(WEIGHTS_DIR, "efficientnet.weights.h5"), 
            data_augmentation
        )
        history_path = os.path.join(HISTORY_DIR, "efficientnet_training_history.csv")
    elif model_name == "mobilenet":
        model = load_mobilenet_model(
            IMG_SIZE, 
            os.path.join(WEIGHTS_DIR, "mobilenet.weights.h5"), 
            data_augmentation
        )
        history_path = os.path.join(HISTORY_DIR, "mobilenet_training_history.csv")
    elif model_name == "resnet":
        model = load_resnet_model(
            IMG_SIZE, 
            os.path.join(WEIGHTS_DIR, "resnet.weights.h5"), 
            data_augmentation
        )
        history_path = os.path.join(HISTORY_DIR, "resnet_training_history.csv")
    elif model_name == "vgg16":
        model = load_vgg_model(
            IMG_SIZE, 
            os.path.join(WEIGHTS_DIR, "vgg16.weights.h5"), 
            data_augmentation
        )
        history_path = os.path.join(HISTORY_DIR, "vgg16_training_history.csv")
    else:
        raise ValueError("Invalid model name. Choose from: 'efficientnet', 'mobilenet', 'resnet', 'vgg16'.")

    return model, history_path

def evaluate_all_models():
    """
    Evaluate all models and print their FLOPs and number of parameters in a formatted table.
    """
    models = ["efficientnet", "mobilenet", "resnet", "vgg16"]
    results = []

    print("\n" + "=" * 75)
    print("{:<15} {:<20} {:<20}".format("Model", "FLOPs", "Parameters"))
    print("{:<15} {:<20} {:<20}".format("", "(GFLOPs)", "(M Params)"))
    print("=" * 75)

    for model_name in models:
        print(f"\nLoading {model_name.capitalize()} model...")
        model, history_path = load_model(model_name)

        # Compute FLOPs
        flops = get_flops(model)

        # Get the number of parameters (convert to millions)
        num_params = model.count_params() / 1e6  

        results.append([model_name.capitalize(), flops, num_params])

        print(f"{model_name.capitalize()} FLOPs: {flops:.2f} GFLOPs, Parameters: {num_params:.2f} M Params")

    # Print results in table format with units in header
    print("\n" + "=" * 75)
    for result in results:
       print("{:<15} {:<15.2f} GFLOPs  {:<15.2f} M Params".format(result[0], result[1], result[2]))
    print("=" * 75 + "\n")



if __name__ == "__main__":
    print("Select a model to evaluate:")
    print("1. EfficientNet")
    print("2. MobileNet")
    print("3. ResNet")
    print("4. VGG16")
    print("5. Evaluate all models (FLOPs & Number of Params)")
    
    try:
        choice = int(input("Enter the number of the model you want to evaluate: "))
    except ValueError:
        print("Invalid input. Please enter a number (1-5).")
        sys.exit(1)

    if choice == 5:
        evaluate_all_models()
        sys.exit(0)

    elif choice == 1:
        model_name = "efficientnet"
    elif choice == 2:
        model_name = "mobilenet"
    elif choice == 3:
        model_name = "resnet"
    elif choice == 4:
        model_name = "vgg16"
    else:
        print("Invalid choice. Please select a valid model (1-5).")
        sys.exit(1)

    print(f"Loading {model_name.capitalize()} model...")
    model, history_path = load_model(model_name)

    # Compute FLOPs
    print(f"Computing FLOPs for {model_name.capitalize()} model...")
    flops = get_flops(model)
    print(f"{model_name.capitalize()} FLOPs: {flops:.2f} GFLOPs")

    # Load training history
    print(f"Loading training history from {history_path}...")
    history_df = load_training_history(history_path)

    # Plot training and validation accuracy/loss
    print("Plotting training and validation accuracy/loss...")
    plot_accuracy_from_csv(history_df)

    # Evaluate the model
    print(f"Evaluating {model_name.capitalize()} model...")
    evaluate_model(model, test_ds, class_names)

    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(model, test_ds, class_names)

    print(f"{model_name.capitalize()} model evaluation and plots completed.")
