import cv2
import numpy as np
from matplotlib import pyplot as plt

def imshow_resized(window_name, img):
    window_size = (int(848), int(480))
    img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, img)


def display_histogram(filename):
    cap = cv2.VideoCapture(filename)

    bins = 16

    # Initialize plot.
    fig, ax = plt.subplots()
    ax.set_title('Histogram (RGB Gray)')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Frequency')

    # Initialize plot line object(s). Turn on interactive plotting and show plot.
    alpha = 0.5
    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=1, alpha=alpha)
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=1, alpha=alpha)
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=1, alpha=alpha)
    lineGray, = ax.plot(np.arange(bins), np.zeros((bins, )), c='k', lw=3)
    ax.set_xlim(0, bins - 1)
    ax.set_ylim(0, 1)
    # Turn on interactive plotting, allowing code to be run while the plot is open
    plt.ion()
    plt.show()

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            imshow_resized('frame', frame)

            numPixels = np.prod(frame.shape[:2])

            (b, g, r) = cv2.split(frame)

            histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
            histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
            histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels

            lineR.set_ydata(histogramR)
            lineG.set_ydata(histogramG)
            lineB.set_ydata(histogramB)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
            lineGray.set_ydata(histogram)

            fig.canvas.draw()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    display_histogram('thailand_vid.mp4')