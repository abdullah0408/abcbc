import tensorflow as tf
import pandas as pd
from src.data_loader import get_datasets
from src.model_tl import build_transfer_model

if __name__ == '__main__':
    train_ds, val_ds = get_datasets('data/metadata.csv', 'data/images')
    num_classes = len(pd.read_csv('data/metadata.csv')['label'].unique())
    model = build_transfer_model(num_classes=num_classes, trainable_layers=30)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)