import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import save_training_history, compile_model, plot_accuracy, plot_confusion_matrix, evaluate_model
from models.efficientnet_model import create_efficientnet_model
from models.mobilenet_model import create_mobilenet_model
from models.resnet_model import create_resnet_model
from models.vgg_model import create_vgg_model
import sys
import os

# Paths
DIR_TRAIN = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/dataset/train"
DIR_VAL = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/dataset/val"
DIR_TEST = "/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/dataset/test"

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Data Preprocessing & Augmentation
train_ds = image_dataset_from_directory(
    DIR_TRAIN, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = image_dataset_from_directory(
    DIR_VAL, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
test_ds = image_dataset_from_directory(
    DIR_TEST, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

# Retrieve class names from dataset
class_names = test_ds.class_names  

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical")
])

def train_model(model_name):
    if model_name == "efficientnet":
        model = create_efficientnet_model(IMG_SIZE, data_augmentation)
    elif model_name == "mobilenet":
        model = create_mobilenet_model(IMG_SIZE, data_augmentation)
    elif model_name == "resnet":
        model = create_resnet_model(IMG_SIZE, data_augmentation)
    elif model_name == "vgg16":
        model = create_vgg_model(IMG_SIZE, data_augmentation)
    else:
        raise ValueError("Invalid model name. Choose from: 'efficientnet', 'mobilenet', 'resnet', 'vgg16'.")

    compile_model(model)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Train model
    print(f"Training {model_name.capitalize()} model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr]
    )
    HISTORY_DIR = "training_histories"
    save_training_history(history, os.path.join(HISTORY_DIR, f"{model_name}_training_history.csv"))

    # Save weights in the "weights" folder
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)  # Create the directory if it doesn't exist
    weights_path = os.path.join(weights_dir, f"{model_name}.weights.h5")
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")

    # Plot training and validation accuracy/loss
    plot_accuracy(history)

    # Evaluate model
    print(f"Evaluating {model_name.capitalize()} model...")
    class_names = test_ds.class_names
    evaluate_model(model, test_ds, class_names)

    # Plot confusion matrix
    plot_confusion_matrix(model, test_ds, class_names)
    print(f"{model_name.capitalize()} model evaluation and plots completed.")


if __name__ == "__main__":
    print("Select a model to train:")
    print("1. EfficientNet")
    print("2. MobileNet")
    print("3. ResNet")
    print("4. VGG16")
    try:
        choice = int(input("Enter the number of the model you want to train: "))
    except ValueError:
        print("Invalid input. Please enter a number (1-4).")
        sys.exit(1)

    if choice == 1:
        train_model("efficientnet")
    elif choice == 2:
        train_model("mobilenet")
    elif choice == 3:
        train_model("resnet")
    elif choice == 4:
        train_model("vgg16")
    else:
        print("Invalid choice. Please select a valid model (1-4).")
        sys.exit(1)