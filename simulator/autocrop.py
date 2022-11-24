import os
import cv2
from os import listdir
from os.path import isfile, join

crop_directory = "/home/jovyan/dso-sr-project-1/simulator/cropped_images"
image_directory = "/home/jovyan/dso-sr-project-1/simulator/real_data"
y = 400
x = 0
h = 800
w = 800
x1 = 0
list1 = []
os.chdir(crop_directory)
u = 0
onlyfiles = [f for f in listdir(image_directory) if isfile(join(image_directory, f))]
onlyfiles.sort()
box = (x, w, y, h)
x = 0
for i in range(0, 10): 
    
    image = cv2.imread("/home/jovyan/dso-sr-project-1/simulator/real_data/" + onlyfiles[i])

    crop_image = image[x:w, y:h]
    
    cv2.imwrite("radar_crop_"+str(x1)+".png", crop_image)
    x1 += 1
    
    cv2.waitKey(0)
        
print(list1)
    
    







