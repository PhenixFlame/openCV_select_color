import cv2
import numpy as np
import pandas as pd

# path_image = "DATA/TRAIN_COLOURS2/color palette.jpg"
# img_path = path_image
# image = cv2.imread(img_path)
# image = cv2.resize(image, (700, 500))

l = range(256)
ll = []
for b in l:
    for g in l:
        for r in l:
            ll.append(b)
            ll.append(g)
            ll.append(r)

nn = np.array(ll, dtype="uint8")
image = nn.reshape(4096, 4096, 3)
cv2.imwrite('/DATA/OUTPUT/image.png', image)
# cv2.namedWindow('color detection')
# cv2.imshow("color detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()