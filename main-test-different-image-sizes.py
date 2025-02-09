import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from utilsimagesize import save_training_history, compile_model, evaluate_model, plot_confusion_matrix, evaluate_compression
from models.mobilenet_model import create_mobilenet_model

# Feedback: Test different image sizes
IMAGE_SIZES = [(224, 224), (128, 128), (64, 64), (32, 32)]
BATCH_SIZE = 32
EPOCHS = 20

# Feedback: Apply Augmentations that are useful
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical")
])


def load_dataset(img_size):
    train_ds = image_dataset_from_directory(
        "dataset/train", label_mode='categorical', image_size=img_size, batch_size=BATCH_SIZE, shuffle=True
    )
    val_ds = image_dataset_from_directory(
        "dataset/val", label_mode='categorical', image_size=img_size, batch_size=BATCH_SIZE, shuffle=False
    )
    test_ds = image_dataset_from_directory(
        "dataset/test", label_mode='categorical', image_size=img_size, batch_size=BATCH_SIZE, shuffle=False
    )
    return train_ds, val_ds, test_ds


def train_model(img_size):
    print(f"\n Training MobileNetV2 model with image size {img_size}...")

    train_ds, val_ds, test_ds = load_dataset(img_size)
    model = create_mobilenet_model(img_size, data_augmentation)
    compile_model(model)

    # Train model
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Save Weights & Training History
    os.makedirs("training_histories", exist_ok=True)
    save_training_history(history, f"training_histories/mobilenet_{img_size[0]}x{img_size[1]}.csv")

    os.makedirs("weights", exist_ok=True)
    model.save_weights(f"weights/mobilenet_{img_size[0]}x{img_size[1]}.h5")

    print(f"\n Model training complete for {img_size}!")

    # Evaluate Model
    evaluation_metrics = evaluate_model(model, test_ds, test_ds.class_names)
    plot_confusion_matrix(model, test_ds, test_ds.class_names)

    # Evaluate Accuracy Before & After Compression
    evaluate_compression(model, img_size[0], test_ds, test_ds.class_names)

    return evaluation_metrics


def train_all_sizes():
    results = []

    for img_size in IMAGE_SIZES:
        metrics = train_model(img_size)
        results.append([
            img_size[0], metrics['accuracy'], metrics['precision'], 
            metrics['recall'], metrics['f1-score'], metrics['mcc']
        ])

    # Create Summary Table
    df = pd.DataFrame(results, columns=["Image Size", "Accuracy", "Precision", "Recall", "F1-Score", "MCC"])
    
    print("\n MobileNetV2 Evaluation Metrics Across Different Image Sizes:")
    print(df.to_string(index=False))  # Print table in terminal

    # Save Summary to CSV
    df.to_csv("training_histories/mobilenet_size_comparison.csv", index=False)
    print(f"\nSummary saved at training_histories/mobilenet_size_comparison.csv")

if __name__ == "__main__":
    train_all_sizes()
