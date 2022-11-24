import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
import cropped_images


## TODO
"""
1. Clean up the data
    - Crop the images --DONE
    - Create labels for the images, by reading off the number of targets
    - Create seperate labels for the images, based on their classes in the image
    - Seperate the images to be used in training or validation
2. Write a new layer of code for another neural network to identify the number of targets
3. Handle the neural network / data such that the first layer of the neural network system will learn how to identify
the number of targets
4. Test and evaluate the model
5. Make any adjustments to the model if needed
6. Submit the final product
"""



"""class car:
    def __init__(self, coordinates, speed):
        self.coordinates = coordinates
        self.velocity = speed

class bicycle:
    def __init__(self, coordinates, speed):
        self.coordinates = coordinates
        self.velocity = speed
class person:
    def __init__(self, coordinates, speed):
        self.coordinates = coordinates
        self.speed = speed

class bus:
    def __init__(self, coordinates, speed):
        self.coordinates = coordinates
        self.velocity = speed
class truck:
    def __init__ (self, coordinates, speed):
        self.coordinates = coordinates
        self.velocity = speed
class motorcycle:
    def __init__(self, coordinates, speed):
        self.coordinates = coordinates
        self.velocity = speed

"""
pixel_values = []
r = 0












all_classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck" ] ##TODO

batch_size = len(onlyfiles = [f for f in listdir(image_directory) if isfile(join(image_directory, f))])
img_height = 800
img_width = 400
import pathlib 
#handling the input images such that 
for i in range(0, batch_size):
    r =  i
    file = 'radar_crop_' + str(r)
    img1.convert('L')
    img = cv2.imread('/home/jovyan/dso-sr-project/simulator/cropped_images/' + file, 0)
    for l in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_values.append(img[l][j])
    

#Labelling the data to use to train the machine    
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.1,
    subset = 'training',
    seed = 123,
    image_size = (image_height, image_width),
    batch_size=batch_size
)

#Labelling the data to be used to validate the machine's accuracy
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.1,
    subset = 'validation',
    seed = 123,
    image_size = (image_height, image_width),
    batch_size = batch_size
)

#building the convolutional neural network based on off the dimensions of the input images (400, 800)
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

#defining the function to train the neural network based on the different params that we set for learning rate
def train_model(learning_rate):
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=my_learning_rate), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
    train = model.fit(train_images, train_labels, epochs = 50,
                  validation_data = (test_images, test_labels))

#plotting the accuracy and the loss of the neural network as it trains.
def plot_accuracy_and_loss():
    plt.plot(train.train['accuracy'], label = 'accuracy')
    plt.plot(train.train['val_accuracy'], label = 'val_accuracy')
    plt.plot(train.train['loss'], label = ['loss'])
    plt.plot(train.train['val_loss'], label ='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1)
    plt.legend(loc = 'lower_right')

#defining the funciton to test the model's accuracy
def test_model():
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose = 2)

    print("test loss" + test_loss, '\n' , "test accuracy" + test_accuracy)


