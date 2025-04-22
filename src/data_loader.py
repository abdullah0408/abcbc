import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_metadata(csv_path):
    return pd.read_csv(csv_path)

def preprocess_image(path, label, img_size=(224,224)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image /= 255.0
    return image, label

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    return image, label

def get_datasets(csv_path, img_dir, batch_size=32, img_size=(224,224)):
    df = load_metadata(csv_path)
    df['path'] = df['image_id'].apply(lambda x: f"{img_dir}/{x}.jpg")
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    # Training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_df['path'], train_df['label']))
    train_ds = train_ds.map(lambda p,l: preprocess_image(p, l, img_size), tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment_image, tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((val_df['path'], val_df['label']))
    val_ds = val_ds.map(lambda p,l: preprocess_image(p, l, img_size), tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds