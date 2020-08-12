"""
Michael Patel
August 2020

Project description:

File description:

conda activate Anomaly-Detection
"""
################################################################################
# Imports
from parameters import *


################################################################################
# autoencoder
def build_autoencoder():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        padding="same",
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Reshape(
        target_shape=(7, 7, 128)
    ))

    model.add(tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.Conv2DTranspose(
        filters=32,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=(3, 3),
        padding="same",
        activation=tf.keras.activations.relu
    ))

    return model
