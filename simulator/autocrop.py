import os
import cv2
import glob
crop_directory = r"C:\Users\eugen\Documents\GitHub\dso-sr-project\simulator\cropped_images"
image_directory = r"C:\Users\eugen\Documents\GitHub\dso-sr-project\simulator\real_data"
y = 400
x = 0
h = 800
w = 800
list1 = []
os.chdir(crop_directory)
u = 0
for i in glob.glob(image_directory):
    u+= 1
    list1.append(i)
    image = cv2.imread(i)
    crop = image[x:w, y:h]
    cv2_imshow(crop)
    cv2.imwrite("vis_radar_crop_" + str(u) + ".png", crop)




