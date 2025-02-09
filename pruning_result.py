import os

def compare_model_compression(unoptimized_model_path, pruned_model_path):
    """
    Compares the size of the unoptimized and pruned VGG16 models and computes the compression rate.

    Args:
        unoptimized_model_path (str): Path to the unoptimized model file.
        pruned_model_path (str): Path to the pruned model file.

    Returns:
        None
    """
    # Ensure both model files exist
    if not os.path.exists(unoptimized_model_path):
        raise FileNotFoundError(f"Unoptimized model not found: {unoptimized_model_path}")
    if not os.path.exists(pruned_model_path):
        raise FileNotFoundError(f"Pruned model not found: {pruned_model_path}")

    # Get model sizes in bytes
    unoptimized_size = os.path.getsize(unoptimized_model_path)
    pruned_size = os.path.getsize(pruned_model_path)

    # Calculate compression ratio and percentage
    compression_ratio = pruned_size / unoptimized_size
    compression_percentage = 100 - (compression_ratio * 100)

    # Display results
    print(f"Unoptimized Model Size: {unoptimized_size / (1024 * 1024):.2f} MB")
    print(f"Pruned Model Size: {pruned_size / (1024 * 1024):.2f} MB")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Compression Percentage: {compression_percentage:.2f}%")


# Define model paths
unoptimized_model_path = "weights/vgg16.weights.h5"
pruned_model_path = "weights/pruned_vgg16.weights.h5"

# Compare compression
compare_model_compression(unoptimized_model_path, pruned_model_path)