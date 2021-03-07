import cv2
import numpy as np
from time import asctime
import funcsource as fs

NPZERO = np.zeros((500, 500, 3))

# COLOURS
GREEN = (0, 255, 0)


class BaseImage:
    name: str
    image_view = None
    image_sample = None
    drawing = False
    clicks = {"DOWN": (0, 0), "UP": (100, 100)}
    REGION_COLOR = GREEN

    def __init__(self, name, image=NPZERO):
        self.image_view = self.image_sample = image
        self.name = name
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, self.mouse_reaction)

    def draw_region(self, p1, p2):
        self.image_view = cv2.rectangle(self.image_sample.copy(), p1, p2, self.REGION_COLOR, 1)

    def mouse_get_region(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.draw_region(self.clicks["DOWN"], (x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks["DOWN"] = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # images["VIEW"] = sample_.copy()
            self.clicks["UP"] = (x, y)
            return self.clicks["DOWN"], self.clicks['UP']

    def work(self, data):
        pass

    def mouse_reaction(self, event, x, y, flags, param):
        data = self.mouse_get_region(event, x, y, flags, param)
        if data is not None:
            self.work(data)

    def show(self):
        cv2.imshow(self.name, self.image_view)

    def update(self, image):
        self.image_view = self.image_sample = cv2.resize(image, (HEIGHT, WIDTH))

    def __repr__(self):
        return f"IMAGE: [{self.name}]"


class SampleView(BaseImage):

    def work(self, data):
        slices = zip(*data)
        xs, ys = map(lambda x: slice(*sorted(x)), slices)  # in numpy: numpy[y][x]

        crop = self.image_sample[ys, xs]
        WINDOWS["CROP"].update(crop.copy())


class CROPView(BaseImage):

    def work(self, data):
        slices = zip(*data)
        xs, ys = map(lambda x: slice(*sorted(x)), slices)  # in numpy: numpy[y][x]
        selected_color = self.image_sample[ys, xs]

        all_colours = zip(*(h for i in selected_color for h in i))
        xx = (set(i) - {0} for i in all_colours)
        min_max = zip(*((min(i), max(i)) for i in xx))

        lower_colour, upper_colour = map(lambda c: np.array(c, dtype="uint8"), min_max)

        sample = WINDOWS["SAMPLE"].image_sample
        mask = cv2.inRange(sample, lower_colour, upper_colour)
        collage = cv2.bitwise_and(sample, sample, mask=mask)

        WINDOWS["TRUE_MASK"].update(collage)


"""


        all_colours = zip(*(h for i in selected_color for h in i))
        xx = (set(i) - {0} for i in all_colours)
        min_max = zip(*((min(i), max(i)) for i in xx))
        lower_colour, upper_colour = map(lambda c: np.array(c, dtype="uint8"), min_max)
        # the mask
        # map(lambda c: np.array(c, dtype="uint8"),
        # mask = cv2.inRange(sample_, lower_colour, upper_colour)
        mask = cv2.inRange(self.image_sample, lower_colour, upper_colour)
        return cv2.bitwise_and(self.image_sample, self.image_sample, mask=mask)
        
        
"""


path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/TRAIN1/2021-03-03 16-25-15.JPG"
# path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/TRAIN1/SAMPLES/CROP_002_IMGA0073.png"

IMAGE = cv2.imread(path_image)

resize_k = round(700 / max(IMAGE.shape), 3)

HEIGHT, WIDTH = map(lambda x: round(resize_k * x), IMAGE.shape[:-1])
IMAGE = cv2.resize(IMAGE, (HEIGHT, WIDTH))

WINDOWS = {
    "SAMPLE": SampleView("SAMPLE", IMAGE),
    "CROP": CROPView("CROP"),
    "TRUE_MASK": CROPView("TRUE_MASK"),
    "FALSE_MASK": CROPView("FALSE_MASK"),
}

while True:

    for image in WINDOWS.values():
        with fs.catch_exceptions(message=image.name):
            image.show()

    key = cv2.waitKey(20)  # wait key in ms
    if key != -1:
        print(key)

    if key == ord('s'):
        for name, image in WINDOWS.items():
            cv2.imwrite(
                f'/home/anfenix/DATA/GIT/OpenCV/DATA/OUTPUT/SAVE/Util/{asctime()}/{name}.jpg',
                image.image_view
            )

    if key & 0xFF in (27, ord('q')):
        break

cv2.destroyAllWindows()




