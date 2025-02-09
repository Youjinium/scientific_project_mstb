from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from utils import create_model
import tensorflow_model_optimization as tfmot

def create_mobilenet_model(img_size, data_augmentation):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    return create_model(base_model, mobilenet_preprocess, img_size, data_augmentation)

def load_mobilenet_model(img_size, weights_path, data_augmentation):
    """
    Loads the MobileNetV2 model with specified weights.

    Parameters:
    - img_size: Tuple, the input image dimensions (height, width).
    - weights_path: Path to the pre-trained weights file.
    - data_augmentation: A `tf.keras.Sequential` object for data augmentation.

    Returns:
    - A compiled `tf.keras.Model` with MobileNetV2 architecture and loaded weights.
    """
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(img_size[0], img_size[1], 3))
    model = create_model(base_model, mobilenet_preprocess, img_size, data_augmentation)
    model.load_weights(weights_path)
    return model

def create_pruned_mobilenet_model(img_size, data_augmentation):
    """
    Creates a pruned version of the MobileNetV2 model.

    Parameters:
    - img_size: Tuple, the input image dimensions (height, width).
    - data_augmentation: A `tf.keras.Sequential` object for data augmentation.

    Returns:
    - A compiled `tf.keras.Model` with pruned MobileNetV2 architecture.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.20, final_sparsity=0.80, begin_step=0, end_step=1000
        )
    }
    pruned_base_model = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
    
    model = create_model(pruned_base_model, mobilenet_preprocess, img_size, data_augmentation)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
    return model, pruning_callback