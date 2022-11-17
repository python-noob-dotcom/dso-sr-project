import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft, ifft, fftshift, ifftshift

from main import data
#before modifying the code for the ml, need to know the number of units and stuff
"""Rv map size is 124,416 pixels large"""
## TODO
"""Create a model
Testing the model
"""

def build_model(learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units = 20, input_shape = "?"))
