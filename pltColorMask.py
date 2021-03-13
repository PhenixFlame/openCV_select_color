import cv2
import numpy as np
from time import asctime
import funcsource as fs
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from plt_rectangle import DrawRectangle
from copy import deepcopy


NPZERO = np.zeros((500, 500, 3))

# COLOURS
GREEN = (0, 255, 0)

MOUSEMOVE = 0
LBUTTONDOWN = 10
LBUTTONUP = 100

MOUSE_EVENTS = {
    ("motion_notify_event", None): MOUSEMOVE,
    ("button_press_event", MouseButton.LEFT): LBUTTONDOWN,
    ("button_release_event", MouseButton.LEFT): LBUTTONUP,
}

ll = ['name',
      'dblclick', 'button', 'key',
      'xdata', 'ydata',
      'x', 'y',
      'inaxes',
      'step',
      'guiEvent']


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


def get_event_info(event):
    event_type = event.name, event.button and int(event.button)
    print(event_type, " : ", MOUSE_EVENTS.get(event_type))
    event_type = MOUSE_EVENTS.get(event_type)
    return event_type, event.xdata, event.ydata


class ViewImage:
    name: str
    image = None
    image_sample = None
    drawing = False
    clicks = {"DOWN": (0, 0), "UP": (100, 100)}
    REGION_COLOR = GREEN

    def __init__(self, name, work_task=None, image=NPZERO):
        self.image = image
        self.name = name
        self.figure = plt.figure()
        self.figure.canvas.set_window_title(name)
        self.axes = self.figure.add_subplot(1, 1, 1)

        if work_task is not None:
            self.work = work_task

        self.cidpress = self.figure.canvas.mpl_connect("button_press_event", self.draw_region)
        self.rectangles = None

    def draw_region(self, event):
        if int(event.button) == 3:
            if self.axes.patches:
                del self.axes.patches[0]
            r = DrawRectangle(event, work=lambda data: self.work(self, data))
            self.rectangles = r
            self.axes.add_patch(r.rect)
            r.connect()

    def work(self, data):
        print(data)

    def show(self):
        self.axes.imshow(self.image)
        self.axes.figure.canvas.draw()

    def update(self, image):
        self.image = image
        self.show()

    def __repr__(self):
        return f"IMAGE: [{self.name}]"


def crop(self, data):
    slices = zip(*data)
    xs, ys = map(lambda x: slice(*sorted(x)), slices)  # in numpy: numpy[y][x]

    crop = self.image[ys, xs]
    WINDOWS["CROP"].update(crop.copy())


def select_color(self, data):
    slices = zip(*data)
    xs, ys = map(lambda x: slice(*sorted(x)), slices)  # in numpy: numpy[y][x]
    selected_color = self.image[ys, xs]

    all_colours = zip(*(h for i in selected_color for h in i))
    xx = (set(i) - {0} for i in all_colours)
    min_max = zip(*((min(i), max(i)) for i in xx))

    lower_colour, upper_colour = map(lambda c: np.array(c, dtype="uint8"), min_max)

    sample = WINDOWS["SAMPLE"].image
    mask = cv2.inRange(sample, lower_colour, upper_colour)
    collage = deepcopy(cv2.bitwise_and(sample, sample, mask=mask))

    # for white backgorund
    white_mask = np.full(collage.shape, 255, dtype=np.uint8)
    # collage = cv2.bitwise_or(collage, white_mask)
    white_collage = np.where(collage[:, :] == [0, 0, 0], white_mask, collage)

    WINDOWS["TRUE_MASK"].update(white_collage.copy())


def deselect_color(self, data):
    pass


path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/TRAIN1/2021-03-03 16-25-15.JPG"
# path_image = "/home/anfenix/DATA/GIT/OpenCV/DATA/TRAIN1/SAMPLES/CROP_002_IMGA0073.png"

IMAGE = cv2.imread(path_image)
IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)


def myprint(data):
    print('hwllo')
    print(data)


WINDOWS = {
    "SAMPLE": ViewImage("SAMPLE", crop, image=IMAGE),
    "CROP": ViewImage("CROP", select_color),
    # "CROP": ViewImage("CROP", myprint),
    "TRUE_MASK": ViewImage("TRUE_MASK", deselect_color),
    # "FALSE_MASK": ViewImage("FALSE_MASK"),
}

for image in WINDOWS.values():
    with fs.catch_exceptions(message=image.name):
        image.show()
plt.show()
