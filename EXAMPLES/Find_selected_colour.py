import cv2
import numpy as np
from time import asctime

# path_image = "DATA/TRAIN_COLOURS2/color palette.jpg"
# path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/TRAIN2/TEST.png"
path_image = "/DATA/TRAIN1/SAMPLES/CROP_002_IMGA0073.png"
# path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/OUTPUT/image.png"
# path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/OUTPUT/image2.jpg"

IMAGE = cv2.imread(path_image)

resize_k = round(700 / max(IMAGE.shape))

HEIGHT, WIDTH = resize_k * IMAGE.shape[1], resize_k * IMAGE.shape[0]
IMAGE = cv2.resize(IMAGE, (HEIGHT, WIDTH))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TEXT_COLOUR = WHITE
COLOUR_DELTA = 15

click = {}
images = {
    "SAMPLE": IMAGE,
    "OUTPUT": IMAGE,
    "VIEW": IMAGE,
    "selected_color": IMAGE[:1, :1]
}
# function to get x,y coordinates of mouse double click

DRAW = False


def draw_function(event, x, y, flags, param):
    global DRAW
    sample_ = images["SAMPLE"]
    if event == cv2.EVENT_MOUSEMOVE and DRAW:
        images["VIEW"] = cv2.rectangle(sample_.copy(), click["DOWN"], (x, y), (0, 255, 0), 1)
    if event == cv2.EVENT_LBUTTONDOWN:
        click["DOWN"] = (x, y)
        DRAW = True
    elif event == cv2.EVENT_LBUTTONUP:
        DRAW = False
        # images["VIEW"] = sample_.copy()
        click["UP"] = (x, y)
        slices = zip(click["UP"], click["DOWN"])
        xs, ys = map(lambda x: slice(*sorted(x)), slices)  # in numpy: numpy[y][x]
        # cv2.imwrite('/home/anfenix/DATA/GIT/OpenCV/DATA/OUTPUT/image2.jpg', image[ss])
        selected_color = sample_[ys, xs]
        images["selected_color"] = selected_color

        all_colours = zip(*(h for i in selected_color for h in i))
        xx = (set(i) - {0} for i in all_colours)
        min_max = zip(*((min(i), max(i)) for i in xx))
        lower_colour, upper_colour = map(lambda c: np.array(c, dtype="uint8"), min_max)
        # the mask
        # map(lambda c: np.array(c, dtype="uint8"),
        # mask = cv2.inRange(sample_, lower_colour, upper_colour)
        mask = cv2.inRange(sample_, lower_colour, upper_colour)
        images["OUTPUT"] = cv2.bitwise_and(sample_, sample_, mask=mask)


cv2.namedWindow('color_detection')
cv2.namedWindow('selected_color')
cv2.setMouseCallback('color_detection', draw_function)

while True:

    cv2.imshow("color_detection", np.hstack([images["VIEW"], images["OUTPUT"]]))
    cv2.imshow("selected_color", images["selected_color"])
    key = cv2.waitKey(20)
    if key != -1:
        print(key)

    if key == ord('s'):
        cv2.imwrite(
            f'/home/anfenix/DATA/GIT/OpenCV/DATA/OUTPUT/SAVE/{asctime()}_image2.jpg',
            np.hstack([images["VIEW"], images["OUTPUT"]])
        )

    if key & 0xFF in (27, ord('q')):
        break

cv2.destroyAllWindows()
