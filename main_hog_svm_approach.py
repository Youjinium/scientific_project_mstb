import os
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


DATASET_PATH = "cvia2"  

# Define Albumentations transformations
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.ChannelShuffle(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Blur(blur_limit=3, p=0.5),
    A.MedianBlur(blur_limit=3, p=0.5),
    A.ToGray(p=0.5),
    A.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5)
])

# Number of times each image is augmented
AUGMENTATION_FACTOR = 5  

def process_image(img):
    """Processes an image by resizing, augmenting, and extracting HOG features."""
    img = cv2.resize(img, (32, 32))
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    augmented_features = []
    for _ in range(AUGMENTATION_FACTOR):
        aug_img = transform(image=img)['image']
        aug_features = hog(aug_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        augmented_features.append(aug_features)
    return features, augmented_features


def load_data(dataset_path):
    """Loads and processes dataset images, applying augmentation and HOG feature extraction."""
    data, labels = [], []
    class_labels = sorted(os.listdir(dataset_path))
    original_count, augmented_count = 0, 0
    feature_vector_size = None
    
    for label, class_name in enumerate(class_labels):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            files = os.listdir(class_path)
            original_count += len(files)
            for file in files:
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    features, augmented_features = process_image(img)
                    if feature_vector_size is None:
                        feature_vector_size = features.shape[0]
                    data.append(features)
                    labels.append(label)
                    data.extend(augmented_features)
                    labels.extend([label] * len(augmented_features))
                    augmented_count += len(augmented_features)
    
    print(f"Total original images: {original_count}")
    print(f"Total images after augmentation: {original_count + augmented_count}")
    print(f"Feature vector size: {feature_vector_size}")
    return np.array(data), np.array(labels)


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_multiclass_roc(y_test, y_score, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = ['red', 'yellow', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def train_and_evaluate(X_train, X_test, y_train, y_test):
    n_classes = len(np.unique(y_train))
    y_train_bin = label_binarize(y_train, classes=range(n_classes))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_score = svm.predict_proba(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)
    plot_multiclass_roc(y_test_bin, y_score, n_classes)


# Load data
X, y = load_data(DATASET_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
print(f"Feature vector size: {X_train.shape[1]}")

# Train and evaluate
train_and_evaluate(X_train, X_test, y_train, y_test)
