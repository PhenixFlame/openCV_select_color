import cv2
import numpy as np

path_image = "DATA/TRAIN_COLOURS2/color palette.jpg"
# path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/OUTPUT/image.png"
# path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/OUTPUT/image2.jpg"

IMAGE = cv2.imread(path_image)
HEIGHT, WIDTH = (700, 500)

IMAGE = cv2.resize(IMAGE, (HEIGHT, WIDTH))


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TEXT_COLOUR = WHITE
COLOUR_DELTA = 15

click = {}
images = {
    "SAMPLE": IMAGE,
    "OUTPUT": IMAGE,
    "selected_color": IMAGE[:1, :1]
}
# function to get x,y coordinates of mouse double click


def draw_function(event, x, y, flags, param):
    sample_ = images["SAMPLE"]
    if event == cv2.EVENT_LBUTTONDOWN:
        click["DOWN"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        click["UP"] = (x, y)
        ss = (slice(click["DOWN"][1], click["UP"][1]), slice(click["DOWN"][0], click["UP"][0])) # in numpy: numpy[y][x]
        # cv2.imwrite('/home/anfenix/DATA/GIT/OpenCV/DATA/OUTPUT/image2.jpg', image[ss])
        selected_color = sample_[ss]
        images["selected_color"] = selected_color

        all_colours = zip(*(h for i in selected_color for h in i))
        xx = (set(i) - {0} for i in all_colours)
        min_max = zip(*((min(i), max(i)) for i in xx))
        lower_colour, upper_colour = map(lambda c: np.array(c, dtype="uint8"), min_max)
        # the mask
        # map(lambda c: np.array(c, dtype="uint8"),
        mask = cv2.inRange(sample_, lower_colour, upper_colour)
        images["OUTPUT"] = cv2.bitwise_and(sample_, sample_, mask=mask)


cv2.namedWindow('color_detection')
cv2.namedWindow('selected_color')
cv2.setMouseCallback('color_detection', draw_function)

while True:

    cv2.imshow("color_detection", np.hstack([images["SAMPLE"], images["OUTPUT"]]))
    cv2.imshow("selected_color", images["selected_color"])
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
