import cv2
import matplotlib.pyplot as plt
import os.path
from os import listdir
import os.path
import numpy as np
from utils import logging

number_directory = "/home/jovyan/dso-sr-project-1/simulator/npy"
plotted_image_directory = "/home/jovyan/dso-sr-project-1/simulator/plotted_rv"
groundtruthdirectory = "/home/jovyan/dso-sr-project-1/simulator/gt"

filenames = [f for f in listdir(number_directory) if os.path.isfile(os.path.join(number_directory, f))]
groundtruthss = [f for f in listdir(groundtruthdirectory) if os.path.isfile(os.path.join(groundtruthdirectory, f))]
filenames.sort()
print(len(filenames))
os.chdir(plotted_image_directory)


for i in range(0, len(filenames)):
    filE = "/home/jovyan/dso-sr-project-1/simulator/npy/" + filenames[i]
    file = np.load(filE)
    plt.imshow(10*np.log10(np.abs(np.sum(file, 1))), cmap = "Blues")
    plt.savefig(filenames[i] + "_plotted.png")
    print(filenames[i] + " finish plotting and saved")
groundtruth = []
for i in range(0, len(filenames)):
    filename = filenames[i]
    a = filenames[i].replace(".npy", ".pickle")
    groundtruth.append(a)

for i in range(0, len(groundtruthss)):
    if groundtruthss[i] in groundtruth:
        pass
    elif groundtruthss[i] not in groundtruth:
        os.chdir(groundtruthdirectory)
        fullfilename = groundtruthdirectory + "/" + groundtruthss[i]
        os.remove(fullfilename)
        print("Removed " + fullfilename)


    
