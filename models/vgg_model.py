from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras import layers, models, regularizers
import tensorflow_model_optimization as tfmot
from utils import create_model


def create_vgg_model(img_size, data_augmentation):
    """
    Creates the VGG16 model with transfer learning.

    Parameters:
    - img_size: Tuple, the input image dimensions (height, width).
    - data_augmentation: A tf.keras.Sequential object for data augmentation.

    Returns:
    - A compiled tf.keras.Model with VGG16 architecture.
    """
    # Load the base VGG16 model pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    
    # Build the model
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = data_augmentation(inputs)  # Apply data augmentation
    x = vgg_preprocess(x)  # Preprocess inputs
    x = base_model(x, training=False)  # Extract features
    x = layers.GlobalAveragePooling2D()(x)  # Global pooling
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  # Fully connected layer
    x = layers.Dropout(0.5)(x)  # Regularization
    outputs = layers.Dense(3, activation='softmax')(x)  # Output for 3 classes
    
    # Create the final model
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_vgg_model(img_size, weights_path, data_augmentation):
    """
    Loads the VGG16 model with specified weights.

    Parameters:
    - img_size: Tuple, the input image dimensions (height, width).
    - weights_path: Path to the pre-trained weights file.
    - data_augmentation: A tf.keras.Sequential object for data augmentation.

    Returns:
    - A compiled tf.keras.Model with VGG16 architecture and loaded weights.
    """
    # Load the base VGG16 model with no pre-trained weights
    base_model = VGG16(weights=None, include_top=False, input_shape=(img_size[0], img_size[1], 3))
    
    # Build the model
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = data_augmentation(inputs)  # Apply data augmentation
    x = vgg_preprocess(x)  # Preprocess inputs
    x = base_model(x, training=False)  # Extract features
    x = layers.GlobalAveragePooling2D()(x)  # Global pooling
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  # Fully connected layer
    x = layers.Dropout(0.5)(x)  # Regularization
    outputs = layers.Dense(3, activation='softmax')(x)  # Output for 3 classes
    
    # Create the final model
    model = models.Model(inputs, outputs)
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_pruned_vgg_model(img_size, data_augmentation):
    """
    Creates a pruned version of the VGG16 model.

    Parameters:
    - img_size: Tuple, the input image dimensions (height, width).
    - data_augmentation: A tf.keras.Sequential object for data augmentation.

    Returns:
    - A compiled tf.keras.Model with pruned VGG16 architecture.
    """
    # Load the base VGG16 model pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # Prune the layers
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.20, final_sparsity=0.80, begin_step=0, end_step=1000
        )
    }
    pruned_base_model = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)

    # Build the pruned model
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = data_augmentation(inputs)
    x = vgg_preprocess(x)
    x = pruned_base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation='softmax')(x)

    # Create the final pruned model
    pruned_model = models.Model(inputs, outputs)
    pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Add pruning step callback
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
    return pruned_model, pruning_callback

def load_pruned_vgg_model(img_size, weights_path, data_augmentation):
    """
    Loads the pruned VGG16 model with specified weights.

    Parameters:
    - img_size: Tuple, the input image dimensions (height, width).
    - weights_path: Path to the pre-trained weights file.
    - data_augmentation: A tf.keras.Sequential object for data augmentation.

    Returns:
    - A compiled tf.keras.Model with pruned VGG16 architecture and loaded weights.
    """
    # Load the base VGG16 model with no pre-trained weights
    base_model = VGG16(weights=None, include_top=False, input_shape=(img_size[0], img_size[1], 3))
    
    # Prune the layers
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.20, final_sparsity=0.80, begin_step=0, end_step=1000
        )
    }
    pruned_base_model = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
    
    # Build the pruned model
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = data_augmentation(inputs)
    x = vgg_preprocess(x)
    x = pruned_base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    # Create the final pruned model
    pruned_model = models.Model(inputs, outputs)

    # Strip pruning before loading weights
    stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    # Load stripped model's weights
    stripped_model.load_weights(weights_path)

    # Compile the model
    stripped_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return stripped_model




















from tensorflow.keras.applications import VGG16
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, PolynomialDecay
from utils import create_model

def old(img_size, data_augmentation):
    """
    Creates a pruned version of the VGG16 model with transfer learning.

    Parameters:
    - img_size: Tuple, the input image dimensions (height, width).
    - data_augmentation: A `tf.keras.Sequential` object for data augmentation.

    Returns:
    - A pruned `tf.keras.Model` with VGG16 architecture.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    
    # Apply pruning to the base model
    pruning_params = {
        'pruning_schedule': PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=1000)
    }
    pruned_base_model = prune_low_magnitude(base_model, **pruning_params)

    # Add the classification head
    return create_model(pruned_base_model, vgg_preprocess, img_size, data_augmentation)
