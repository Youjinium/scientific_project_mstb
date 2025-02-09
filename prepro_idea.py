import cv2
import numpy as np
import os


def preprocess_thermal_image(image_path, brightness_threshold=127):
    # Load the image
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]
    
    # Create masks for bright (hot) and cold (not bright) regions
    bright_mask = (value_channel >= brightness_threshold).astype(np.uint8) * 255
    cold_mask = (value_channel < brightness_threshold).astype(np.uint8) * 255
    
    # Calculate percentage of bright and cold regions
    bright_percentage = np.sum(bright_mask > 0) / (image.shape[0] * image.shape[1]) * 100
    cold_percentage = np.sum(cold_mask > 0) / (image.shape[0] * image.shape[1]) * 100
    
    return bright_percentage, cold_percentage

def classify_image(bright_percentage):
    up = 63
    down = 11.5
    if bright_percentage > up:
        return "DEC_FM_VACUUM_ALL_HEATERS"
    elif down <= bright_percentage <= up:
        return "DEC_FM_VACUUM_HEATER_0"
    else:
        return "DEC_FM_VACUUM_NO_HEATERS"

def evaluate_classification(folders):
    correct_predictions = 0
    total_predictions = 0
    wrong_predictions = []  # List to store wrongly predicted images

    for folder in folders:
        true_label = os.path.basename(folder).upper()  # Use folder name as true label
        for img_name in os.listdir(folder):
            if img_name.endswith('.png'):  # Assuming PNG format
                img_path = os.path.join(folder, img_name)
                bright_percentage, cold_percentage = preprocess_thermal_image(img_path)
                predicted_label = classify_image(bright_percentage)
                
                #print(f"Image: {img_name}, Predicted: {predicted_label}, Bright Percentage: {bright_percentage:.2f}%, Cold Percentage: {cold_percentage:.2f}%")
                
                # Check if the prediction is correct
                if predicted_label == true_label:
                    correct_predictions += 1
                else:
                    # Store additional percentage information for wrong predictions
                    wrong_predictions.append((img_name, predicted_label, true_label, bright_percentage, cold_percentage))

                total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\nTotal Predictions: {total_predictions}, Correct Predictions: {correct_predictions}, Accuracy: {accuracy:.2f}%")

    # Print out the wrongly predicted images with percentages
    if wrong_predictions:
        print("\nWrongly Predicted Images:")
        for img_name, predicted_label, true_label, bright_percentage, cold_percentage in wrong_predictions:
            print(f"Image: {img_name}, Predicted: {predicted_label}, True Label: {true_label}, "
                  f"Bright Percentage: {bright_percentage:.2f}%, Cold Percentage: {cold_percentage:.2f}%")

# Specify the folders for each case
folders = [
    '/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/cvia2/DEC_FM_VACUUM_ALL_HEATERS',    # Replace with your actual path
    '/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/cvia2/DEC_FM_VACUUM_HEATER_0',     # Replace with your actual path
    '/home/youji/Documents/Space_Master/ISM_Eugene/Semester 3/Scientific Project/cvia2/DEC_FM_VACUUM_NO_HEATERS'    # Replace with your actual path
]

# Evaluate the classification
evaluate_classification(folders)

