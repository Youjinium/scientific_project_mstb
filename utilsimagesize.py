import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, matthews_corrcoef, roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

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



def save_training_history(history, filename):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(filename, index=False)
    print(f" Training history saved to {filename}")


def evaluate_model(model, test_ds, class_names):
    """
    Evaluates a trained model using:
    - Matthews Correlation Coefficient (MCC)
    - F1-score
    - Precision, Recall
    - Accuracy
    - ROC-AUC Curve
    """

    y_true, y_pred, y_prob = [], [], []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_prob.extend(predictions)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n Classification Report:")
    for label in class_names:
        print(f"  {label}: Precision={report[label]['precision']:.3f}, Recall={report[label]['recall']:.3f}, F1={report[label]['f1-score']:.3f}")

    print(f"\n Matthews Correlation Coefficient (MCC): {mcc:.3f}")
    print(f"Accuracy: {report['accuracy']:.3f}")


    plot_roc_curve(y_true, y_prob, class_names)

    # Return metrics 
    return {
        "accuracy": report["accuracy"],
        "precision": np.mean([report[label]["precision"] for label in class_names]),
        "recall": np.mean([report[label]["recall"] for label in class_names]),
        "f1-score": np.mean([report[label]["f1-score"] for label in class_names]),
        "mcc": mcc
    }


def plot_roc_curve(y_true, y_prob, class_names):
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(np.array(y_true) == i, np.array(y_prob)[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_score:.3f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend()
    plt.show()


def plot_accuracy(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Training & Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Plot loss
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Training & Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.show()


def plot_confusion_matrix(model, test_ds, class_names):
    y_true, y_pred = [], []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def evaluate_compression(model, img_size, test_ds, class_names):
    print("\n Evaluating model BEFORE compression...")
    pre_compression_metrics = evaluate_model(model, test_ds, class_names)

    # Compress Model
    compressed_model_path = f"weights/mobilenet_{img_size}_compressed.tflite"
    convert_to_tflite(model, compressed_model_path)

    print("\n Evaluating model AFTER compression...")
    post_compression_metrics = evaluate_tflite_model(compressed_model_path, test_ds, class_names)

    # Print the accuracy drop comparison
    print("\n Comparison Before & After Compression:")
    print(f"Accuracy Before: {pre_compression_metrics['accuracy']:.3f}")
    print(f"Accuracy After : {post_compression_metrics['accuracy']:.3f}")


def convert_to_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f" Model compressed and saved to {output_path}")


def evaluate_tflite_model(tflite_model_path, test_ds, class_names):
    print(f" Evaluating compressed model from {tflite_model_path}...")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true, y_pred = [], []

    for images, labels in test_ds:
        for i in range(len(images)):
            input_data = np.expand_dims(images[i].numpy(), axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            y_pred.append(np.argmax(output_data))
            y_true.append(np.argmax(labels[i].numpy()))

    print("\n Classification Report (TFLite Model):\n", classification_report(y_true, y_pred, target_names=class_names))
    print("MCC (TFLite):", matthews_corrcoef(y_true, y_pred))

    return {
        "accuracy": classification_report(y_true, y_pred, output_dict=True)["accuracy"],
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
