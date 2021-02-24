import pathlib

import keras
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


def save_model(model_path:str):
    pass

def load_model(model_path:str):
    pass

def load_data(dataset_url)->str:
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    print(data_dir)
    return data_dir

def construct_data(data_dir,batch_size,img_height,img_width,subset)->tf.data.Dataset:
    ds=tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        batch_size=batch_size,
        image_size=(img_height, img_width),
        seed=123,
        validation_split=0.2,
        subset=subset,
    )
    return ds

def construct_model():
    pass

if __name__ == "__main__":
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir=load_data(dataset_url)

    batch_size=32
    img_height,img_width=180,180

    train_ds=construct_data(data_dir,batch_size,img_height,img_width,"training")
    validate_ds=construct_data(data_dir,batch_size,img_height,img_width,"validation")

    class_names=train_ds.class_names
    print(class_names)

