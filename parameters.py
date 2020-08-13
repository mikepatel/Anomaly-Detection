"""
Michael Patel
August 2020

Project description:

File description:

conda activate Anomaly-Detection
"""
################################################################################
# Imports
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


################################################################################
# image dimensions
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1

NUM_EPOCHS = 10
BATCH_SIZE = 64

ANOMALY_LABEL = 9
