import cv2
import numpy as np


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grayscale_17_levels(image):
    high = 255
    while True:
        low = high - 15
        col_low = np.array([low])
        col_high = np.array([high])
        mask = cv2.inRange(gray, col_low, col_high)
        gray[mask > 0] = high
        high -= 15
        if low == 0:
            break


image = cv2.imread('/DATA/TRAIN2/TEST.png')
viewImage(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayscale_17_levels(gray)
viewImage(gray)
