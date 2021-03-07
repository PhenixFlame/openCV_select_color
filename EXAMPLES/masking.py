import cv2
import numpy as np


path_image = "/DATA/TRAIN1/SAMPLES/CROP_002_IMGA0073.png"

img = cv2.imread(path_image)
# cv2.imshow("Wave", img)
# cv2.waitKey(0)
# we readed the image

mask = np.zeros(img.shape[:2], dtype="uint8")
mask[:50, :50] = 0
# (cx, cy) = map(int,(img.shape[1] / 2, img.shape[0] / 2))
# cv2.rectangle(mask, (cx - 75, cy - 75), (cx + 75, cy + 75), 255, 1)
# cv2.rectangle(mask, (0, 0), (50, 50), 255, -1)
cv2.imshow("Mask", mask)
cv2.waitKey(0)

masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Masked", masked)
cv2.waitKey(0)
