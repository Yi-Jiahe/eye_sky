import cv2
import numpy as np
from matplotlib import pyplot as plt


class DynamicHistogram:

    def __init__(self, bins):
        self.bins = bins

        self.fig, self.ax = plt.subplots()

        alpha = 0.5
        self.line_r, = self.ax.plot(np.arange(bins), np.zeros((self.bins,)), c='r', lw=1, alpha=alpha)
        self.line_g, = self.ax.plot(np.arange(bins), np.zeros((self.bins,)), c='g', lw=1, alpha=alpha)
        self.line_b, = self.ax.plot(np.arange(bins), np.zeros((self.bins,)), c='b', lw=1, alpha=alpha)
        self.line_gray, = self.ax.plot(np.arange(bins), np.zeros((self.bins,)), c='k', lw=3)

    def initialize_plot(self):
        # Initialize plot.
        self.ax.set_title('Histogram (RGB Gray)')
        self.ax.set_xlabel('Bin')
        self.ax.set_ylabel('Frequency')

        # Initialize plot line object(s).
        self.ax.set_xlim(0, self.bins - 1)
        self.ax.set_ylim(0, 1)

    def plot(self, frame):
        num_pixels = np.prod(frame.shape[:2])

        (b, g, r) = cv2.split(frame)

        histogram_r = cv2.calcHist([r], [0], None, [self.bins], [0, 255]) / num_pixels
        histogram_g = cv2.calcHist([g], [0], None, [self.bins], [0, 255]) / num_pixels
        histogram_b = cv2.calcHist([b], [0], None, [self.bins], [0, 255]) / num_pixels

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram_gray = cv2.calcHist([gray], [0], None, [self.bins], [0, 255]) / num_pixels

        self.line_r.set_ydata(histogram_r)
        self.line_g.set_ydata(histogram_g)
        self.line_b.set_ydata(histogram_b)
        self.line_gray.set_ydata(histogram_gray)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def imshow_resized(window_name, img):
    window_size = (int(848), int(480))
    img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, img)


def display_histograms(filename):
    cap = cv2.VideoCapture(filename)

    bins = 16

    original = DynamicHistogram(bins)
    adjusted = DynamicHistogram(bins)

    original.initialize_plot()
    adjusted.initialize_plot()

    # Turn on interactive plotting, allowing code to be run while the plot is open
    plt.ion()
    plt.show()

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            imshow_resized('frame', frame)

            original.plot(frame)

            masked = cv2.convertScaleAbs(frame, alpha=2, beta=128)

            imshow_resized('adjusted', masked)

            adjusted.plot(masked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    display_histograms('thailand_vid.mp4')