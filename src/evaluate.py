from tensorflow.keras import models, layers
from tensorflow.keras.applications import MobileNetV2

def build_transfer_model(input_shape=(224,224,3), num_classes=2, trainable_layers=20):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base.layers[:-trainable_layers]:
        layer.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=outputs)