import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft, ifft, fftshift, ifftshift

from main import test_data
#before modifying the code for the ml, need to know the number of units and stuff
"""Rv map size is 124,416 pixels large"""
## TODO
"""Create a model
Testing the model
"""

(train_images, train_labels), (test_images, test_labels) = "file_name_here"

all_classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck" ]


model = tf.keras.model.Sequential()
model.add([
    tf.keras.layers.Conv2D(256, (10, 10), activation = 'relu', input_shape = (256, 256, 64)),
    tf.keras.Maxpooling2D(2, 2),
    tf.keras.layers.Conv2D(512, (10, 10), activation = 'relu'),
    tf.keras.Maxpooling2D(2, 2),
    tf.keras.layers.Conv2D(1024, (10, 10), activation = 'relu'),
    tf.keras.Maxpooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    
])

model.summary()

