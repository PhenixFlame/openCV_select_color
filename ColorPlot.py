from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import cv2
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/TRAIN1/2021-03-03 16-25-15.JPG"
path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/TRAIN1/SAMPLES/CROP_002_IMGA0073.png"
path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/TRAIN_COLORS/rgb_nemo.webp"

nemo = cv2.imread(path_image)
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(nemo)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.add_patch()
pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
rf, gf, bf = r.flatten(), g.flatten(), b.flatten()
axis.scatter(rf, gf, bf, facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(hsv_nemo)
fig = plt.figure()
axis2 = fig.add_subplot(1, 1, 1, projection="3d")


axis2.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis2.set_xlabel("Hue")
axis2.set_ylabel("Saturation")
axis2.set_zlabel("Value")
plt.show()