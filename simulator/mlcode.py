import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from main import test_data
#before modifying the code for the ml, need to know the number of units and stuff
"""Rv map size is 124,416 pixels large"""
## TODO
"""Create a model
Testing the model
"""



all_classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck" ]

batch_size = 100
img_height = 256
img_width = 256
dataset_url = 'https://drive.google.com/drive/folders/14pPvU2FRdLpc8ZS904oE20ZFYfFjJvaM'


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset = 'training',
    seed = 123,
    image_size = (image_height, image_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset = 'validation',
    seed = 123,
    image_size = (image_height, image_width),
    batch_size = batch_size
)


model = tf.keras.model.Sequential()
model.add([
    tf.keras.layers.Conv2D(256, (10, 10), activation = 'relu', input_shape = (256, 256, 64)),
    tf.keras.Maxpooling2D(2, 2),
    tf.keras.layers.Conv2D(512, (10, 10), activation = 'relu'),
    tf.keras.Maxpooling2D(2, 2),
    tf.keras.layers.Conv2D(1024, (10, 10), activation = 'relu'),
    tf.keras.Maxpooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(6, activation = 'relu')
    
])

model.summary()

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

test_loss, test_accuracy = model.evaluate(tets_images, test_labels, verbose = 2)

print("test loss" + test_loss, '\n' , "test accuracy" + test_accuracy)
