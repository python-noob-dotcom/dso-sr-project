from os import listdir
import os
import os.path
from autoplot import filenames

#Code to remove images

crop_image = "/home/jovyan/dso-sr-project-1/simulator/cropped_images"
uncropped_images = "/home/jovyan/dso-sr-project-1/simulator/real_data"

filelist = [f for f in listdir(crop_image) if os.path.isfile(os.path.join(crop_image, f))]
filelist.sort()

for i in range(len(filenames), len(filelist)):
    file = crop_image + "/" + filelist[i]
    os.remove(file)
    print(filelist[i] + " removed from " + uncropped_images)

# auto-rename
os.chdir(uncropped_images)
for i in range(0, len(filelist)):
    os.rename(filelist[i], "vis_" + str(i) + ".png")
    print("Renamed " + "vis_" + str(i) + ".png")

print(len(filenames))
print(str(len(filelist)) + "cropped images number")
