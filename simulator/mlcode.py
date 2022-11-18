import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL







class car:
    def __init__(self, coordinates, speed):
        self.coordinates = coordinates
        self.velocity = speed

class bicycle:
    def __init__(self, coordinates, speed):
        self.coordinates = coordinates
        self.velocity = speed



all_classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck" ] ##TODO

batch_size = 100
img_height = 256
img_width = 256
import pathlib 
dataset_url = "/home/jovyan/dso-sr-project/simulator/data"
data_dir = tf.keras.utils.get_file('vis', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.1,
    subset = 'training',
    seed = 123,
    image_size = (image_height, image_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.1,
    subset = 'validation',
    seed = 123,
    image_size = (image_height, image_width),
    batch_size = batch_size
)

def build_model():
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(256, (10, 10), activation = 'relu', input_shape = (256, 256, 64)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(512, (10, 10), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(1024, (10, 10), activation = 'relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(6, activation = 'relu')])
    model.summary()

    
def train_model():
    model.compile(optimizer = "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
    train = model.fit(train_images, train_labels, epochs = 50,
                  validation_data = (test_images, test_labels))

def plot_accuracy_and_loss():
    plt.plot(train.train['accuracy'], label = 'accuracy')
    plt.plot(train.train['val_accuracy'], label = 'val_accuracy')
    plt.plot(train.train['loss'], label = ['loss'])
    plt.plot(train.train['val_loss'], label ='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1)
    plt.legend(loc = 'lower_right')

def test_model():
    test_loss, test_accuracy = model.evaluate(tets_images, test_labels, verbose = 2)

    print("test loss" + test_loss, '\n' , "test accuracy" + test_accuracy)

build_model()
