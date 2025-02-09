from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from utils import create_model

def create_efficientnet_model(img_size, data_augmentation):
    # The include_top=False removes the fully connected layers, keeping only the feature extractor.
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    return create_model(base_model, efficientnet_preprocess, img_size, data_augmentation)

def load_efficientnet_model(img_size, weights_path, data_augmentation):
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(img_size[0], img_size[1], 3))
    model = create_model(base_model, efficientnet_preprocess, img_size, data_augmentation)
    model.load_weights(weights_path, by_name=True)
    return model
