import matplotlib.pyplot as plt

lll = []


def draw_rectangle(event):
    x, y = event.xdata, event.ydata
    rect = plt.Rectangle((x, y), 0, 0)

    axes.add_patch(rect)
    axes.figure.canvas.draw()
    dr.connect()
    dr = DrawRectangle(rect, (x, y))
    lll.append(dr)  # Important! in add_path week links and after that dr object will be deleted!


class DrawRectangle:
    def __init__(self, event, work):
        self.press_xy = event.xdata, event.ydata
        self.rect = plt.Rectangle(self.press_xy, 0, 0, fill=False, edgecolor='pink', lw=1)
        self.work = work

    def connect(self):
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_motion(self, event):
        x0, y0 = self.press_xy
        dx = event.xdata - x0
        dy = event.ydata - y0
        self.rect.set_width(dx)
        self.rect.set_height(dy)
        self.rect.figure.canvas.draw()

    def on_release(self, event):
        self.on_motion(event)
        self.rect.figure.canvas.draw()
        self.disconnect()
        data = self.press_xy, (event.xdata, event.ydata)
        data = map(lambda x: map(int, x), data)
        self.work(data)

    def disconnect(self):
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)
