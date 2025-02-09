from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from utils import create_model

def create_resnet_model(img_size, data_augmentation):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    return create_model(base_model, resnet_preprocess, img_size, data_augmentation)

def load_resnet_model(img_size, weights_path, data_augmentation):
    """
    Loads the ResNet50 model with specified weights.

    Parameters:
    - img_size: Tuple, the input iamage dimensions (height, width).
    - weights_path: Path to the pre-trained weights file.
    - data_augmentation: A `tf.keras.Sequential` object for data augmentation.

    Returns:
    - A compiled `tf.keras.Model` with ResNet50 architecture and loaded weights.
    """
    base_model = ResNet50(weights=None, include_top=False, input_shape=(img_size[0], img_size[1], 3))
    model = create_model(base_model, resnet_preprocess, img_size, data_augmentation)
    model.load_weights(weights_path)
    return model