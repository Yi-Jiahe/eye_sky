import cv2
import numpy as np
from matplotlib import pyplot as plt


class DynamicHistogram():
    def initialize_plot(self, bins):
        # Initialize plot.
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Histogram (RGB Gray)')
        self.ax.set_xlabel('Bin')
        self.ax.set_ylabel('Frequency')

        # Initialize plot line object(s).
        alpha = 0.5
        self.line_r, = self.ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=1, alpha=alpha)
        self.line_g, = self.ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=1, alpha=alpha)
        self.line_b, = self.ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=1, alpha=alpha)
        self.line_gray, = self.line_gray.plot(np.arange(bins), np.zeros((bins,)), c='k', lw=3)
        self.ax.set_xlim(0, bins - 1)
        self.ax.set_ylim(0, 1)

    def plot(self, frame, bins):
        num_pixels = np.prod(frame.shape[:2])

        (b, g, r) = cv2.split(frame)

        histogram_r = cv2.calcHist([r], [0], None, [bins], [0, 255]) / num_pixels
        histogram_g = cv2.calcHist([g], [0], None, [bins], [0, 255]) / num_pixels
        histogram_b = cv2.calcHist([b], [0], None, [bins], [0, 255]) / num_pixels

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram_gray = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / num_pixels

        self.line_r.set_ydata(histogram_r)
        self.line_g.set_ydata(histogram_g)
        self.line_b.set_ydata(histogram_b)
        self.line_gray.set_ydata(histogram_gray)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()




def imshow_resized(window_name, img):
    window_size = (int(848), int(480))
    img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, img)


def display_histograms(filename):
    cap = cv2.VideoCapture(filename)

    bins = 16

    fig_original, lines_original = initialize_plot(bins)

    fig_adjusted, lines_adjusted = initialize_plot(bins)

    # Turn on interactive plotting, allowing code to be run while the plot is open
    plt.ion()
    plt.show()

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            imshow_resized('frame', frame)

            histograms_original = extract_histograms(frame, bins)

            lines_original[0].set_ydata(histograms_original[0])
            lines_original[1].set_ydata(histograms_original[1])
            lines_original[2].set_ydata(histograms_original[2])
            lines_original[3].set_ydata(histograms_original[3])

            masked = cv2.convertScaleAbs(frame, alpha=2, beta=128)

            imshow_resized('adjusted', masked)

            histograms_adjusted = extract_histograms(masked, bins)

            lines_adjusted[0].set_ydata(histograms_adjusted[0])
            lines_adjusted[1].set_ydata(histograms_adjusted[1])
            lines_adjusted[2].set_ydata(histograms_adjusted[2])
            lines_adjusted[3].set_ydata(histograms_adjusted[3])

            fig_original.canvas.draw()
            fig_original.canvas.flush_events()

            fig_adjusted.canvas.draw()
            fig_adjusted.canvas.flush_events()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def initialize_plot(bins):
    # Initialize plot.
    fig, ax = plt.subplots()
    ax.set_title('Histogram (RGB Gray)')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Frequency')

    # Initialize plot line object(s). Turn on interactive plotting and show plot.
    alpha = 0.5
    line_r, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=1, alpha=alpha)
    line_g, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=1, alpha=alpha)
    line_b, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=1, alpha=alpha)
    line_gray, = ax.plot(np.arange(bins), np.zeros((bins, )), c='k', lw=3)
    ax.set_xlim(0, bins - 1)
    ax.set_ylim(0, 1)

    return fig, [line_r, line_g, line_b, line_gray]


def extract_histograms(frame, bins):
    num_pixels = np.prod(frame.shape[:2])

    (b, g, r) = cv2.split(frame)

    histogram_r = cv2.calcHist([r], [0], None, [bins], [0, 255]) / num_pixels
    histogram_g = cv2.calcHist([g], [0], None, [bins], [0, 255]) / num_pixels
    histogram_b = cv2.calcHist([b], [0], None, [bins], [0, 255]) / num_pixels

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histogram_gray = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / num_pixels

    return [histogram_r, histogram_g, histogram_b, histogram_gray]


if __name__ == '__main__':
    display_histograms('thailand_vid.mp4')