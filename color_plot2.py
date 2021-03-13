import matplotlib.pyplot as plt
import cv2
from matplotlib import colors
import numpy as np

path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/TRAIN1/SAMPLES/CROP_002_IMGA0073.png"

image = cv2.imread(path_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hsv_nemo = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

fig2 = plt.figure()
fig2.canvas.set_window_title("HSV")
axis = fig2.add_subplot(1, 1, 1, projection="3d")


def color_plot(image, axis):
    x, y, z = cv2.split(image)
    w, h, _ = np.shape(image)
    pixel_colors = image.reshape((w * h, 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(x.flatten(), y.flatten(), z.flatten(), facecolors=pixel_colors, marker=".")
    axis.figure.canvas.draw()


def simple_color_plot(image, axis, size=100):
    colours = list(set(tuple(y) for x in image for y in x))
    # colours = list(set(tuple(y) for x in image.round(-1) for y in x))
    x, y, z = map(np.array, zip(*colours))
    # w, h, _ = map(len, (x,y,z))
    pixel_colors = colours
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(x.flatten(), y.flatten(), z.flatten(), s=[size for i in x], facecolors=pixel_colors, marker=".")
    axis.figure.canvas.draw()

