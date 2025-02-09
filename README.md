# Thermal Anomaly Detection in Spacecraft Systems

This project explores deep learning models for detecting thermal anomalies in spacecraft. It implements and optimizes various architectures, including **EfficientNet, MobileNetV2, ResNet, and VGG16**, along with a **HOG + SVM** approach for benchmarking.

## Installation

### 1. Install Dependencies Using pipenv
```bash
pipenv install
pipenb shell
```

## Code Structure

### Model Training & Prediction
- **`main.py`** – Trains **EfficientNet, MobileNetV2, ResNet, and VGG16** models.
- **`main_predict.py`** – Runs predictions using trained models.

### Model Optimization
- **`main_vgg16_optimization.py`** – Applies pruning, quantization, and compression to VGG16.
- **`main_vgg16_predict.py`** – Evaluates the optimized VGG16 model.
- **`main_mobilenetv2_pruning.py`** – Optimizes MobileNetV2.

### Experimental Tests Based on Feedback from Carl
- **`main-test-different-image-sizes.py`** – Tests different image resolutions.
- **`main_hog_svm_approach.py`** – Explores the **HOG + SVM** approach.

### Other
- **`prepro_idea.py`** – Initial experiment for extracting cold and hot thresholds for classification.

## Folder Structure
- **`models/`** – Contains the trained **EfficientNet, MobileNetV2, ResNet, and VGG16** models.
- **`training_histories/`** – Stores CSV files with training results.
- **`weights/`** – Saves trained model weights.

## Dataset Not Included
The dataset is too large to be uploaded to this repository.
