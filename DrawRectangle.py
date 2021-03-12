import numpy as np
import matplotlib.pyplot as plt

lll =[]
def draw_rectangle(event):
    x,y = event.xdata, event.ydata
    rect = plt.Rectangle((x,y), 0,0)

    axes.add_patch(rect)
    axes.figure.canvas.draw()
    dr = DraggableRectangle(rect, (x, y))
    dr.connect()
    lll.append(dr) # Important! in add_path week links and after that dr object will be deleted!

class DraggableRectangle:
    def __init__(self, rect:plt.Rectangle, xy):
        self.rect = rect
        self.press_xy = xy

    def connect(self):
        'connect to all the events we need'
        self.cidrelease = fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = fig.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    # def on_press(self, event):
    #     'on button press we will see if the mouse is over us and store some data'
    #     if event.inaxes != self.rect.axes: return
    #
    #     contains, attrd = self.rect.contains(event)
    #     if not contains: return
    #     print('event contains', self.rect.xy)
    #     x0, y0 = self.rect.xy
    #     self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        print("on_motion")
        # if self.press is None: return
        # if event.inaxes != self.rect.axes: return
        x0, y0 = self.press_xy
        dx = event.xdata - x0
        dy = event.ydata - y0
        #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #      (x0, xpress, event.xdata, dx, x0+dx))
        self.rect.set_width(dx)
        self.rect.set_height(dy)
        self.rect.figure.canvas.draw()

    def on_release(self, event):
        print("on_motion")

        'on release we reset the press data'
        # self.press = None
        self.on_motion(event)
        self.rect.figure.canvas.draw()
        self.disconnect()

    def disconnect(self):
        'disconnect all the stored connection ids'
        # self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)



fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot([0, 10],[0, 10])
rr = plt.Rectangle((1, 1), 2, 6,
             edgecolor = 'pink',
             facecolor = 'blue',
             fill=True,
             lw=5)

axes.add_patch(rr)

fig.canvas.mpl_connect(
    'button_press_event', draw_rectangle)

# rects = ax.bar(range(10), 20*np.random.rand(10))
# drs = []
# for rect in rects:
#     dr = DraggableRectangle(rect)
#     dr.connect()
#     drs.append(dr)

plt.show()