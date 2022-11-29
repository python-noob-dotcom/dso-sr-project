import cv2
import matplotlib.pyplot as plt
import os.path
from os import listdir
import os.path
import numpy as np
import mpmath


number_directory = "/home/jovyan/dso-sr-project-1/simulator/number"
plotted_image_directory = "/home/jovyan/dso-sr-project-1/simulator/plotted_rsv"

filenames = [f for f in listdir(number_directory) if os.path.isfile(os.path.join(number_directory, f))]
filenames.sort()

os.chdir(plotted_image_directory)


for i in range(0, 10):
    filE = "/home/jovyan/dso-sr-project-1/simulator/number/" + filenames[i]
    file = np.load(filE)
    plt.imshow(10*np.log10(np.abs(np.sum(file, 1))))
    plt.savefig(filenames[i] + "_plotted.png")

    