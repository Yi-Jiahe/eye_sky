import cv2
import numpy as np
from matplotlib import pyplot as plt

from camera_stabilizer import Camera

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

    def plot(self, frame, mask=None):
        num_pixels = np.prod(frame.shape[:2])

        if len(frame.shape) == 3:
            # cv2.split() is an expensive function, thus numpy indexing is preferred
            # (b, g, r) = cv2.split(frame)
            b = frame[:, :, 0]
            g = frame[:, :, 1]
            r = frame[:, :, 2]

            histogram_r = cv2.calcHist([r], [0], mask, [self.bins], [0, 256]) / num_pixels
            histogram_g = cv2.calcHist([g], [0], mask, [self.bins], [0, 256]) / num_pixels
            histogram_b = cv2.calcHist([b], [0], mask, [self.bins], [0, 256]) / num_pixels

            self.line_r.set_ydata(histogram_r)
            self.line_g.set_ydata(histogram_g)
            self.line_b.set_ydata(histogram_b)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        else:
            gray = frame

        histogram_gray = cv2.calcHist([gray], [0], mask, [self.bins], [0, 256]) / num_pixels

        self.line_gray.set_ydata(histogram_gray)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def imshow_resized(window_name, img):
    window_size = (int(768), int(432))
    img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, img)


def display_histograms(filename):
    cap = cv2.VideoCapture(filename)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    camera = Camera([frame_width, frame_height])

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
            frame, mask = camera.undistort(frame)

            imshow_resized('frame', frame)

            original.plot(frame, mask)

            # masked = cv2.convertScaleAbs(frame, alpha=2, beta=128)

            masked = threshold_rgb(frame)

            imshow_resized('adjusted', masked)

            adjusted.plot(masked, mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def threshold_rgb(frame, threshold_r=127, threshold_g=127, threshold_b=127):
    # (b, g, r) = cv2.split(frame)
    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]

    ret, thresh_r = cv2.threshold(r, threshold_r, 255, cv2.THRESH_BINARY)
    ret, thresh_g = cv2.threshold(g, threshold_g, 255, cv2.THRESH_BINARY)
    ret, thresh_b = cv2.threshold(b, threshold_b, 255, cv2.THRESH_BINARY)

    # Combine green and blue threshold masks (and red for what its worth)
    # Such that the mask covers all pixels where blue green or red values are above their respective thresholds
    mask = thresh_r | thresh_g | thresh_b
    mask = cv2.bitwise_not(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    frame_inv = cv2.bitwise_not(frame)

    out = cv2.bitwise_and(frame_inv, mask)
    out = cv2.bitwise_not(out)

    return out


if __name__ == '__main__':
    display_histograms('stabilized.mp4')