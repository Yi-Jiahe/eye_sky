from object_tracking_rt import track_objects_realtime, imshow_resized
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import multiprocessing


class TrackPlot():
    def __init__(self, track_id):
        self.id = track_id
        self.xs = np.array([], dtype=int)
        self.ys = np.array([], dtype=int)
        self.frameNos = np.array([], dtype=int)
        self.colourized_times = []
        self.lastSeen = 0

    def plot_track(self):
        print(f"Track {self.id} being plotted...")
        plt.scatter(self.xs, self.ys, c=self.colourized_times, marker='+')
        plt.show()

    def update(self, offset, location, frame_no):
        self.xs = np.append(self.xs, [int(location[0] + offset[0])])
        self.ys = np.append(self.ys, [int(location[1] + offset[1])])
        self.frameNos = np.append(self.frameNos, [frame_no])
        self.lastSeen = frame_no


def plot_tracks_realtime():
    print('b4 loop')

    q = multiprocessing.Queue()

    get_results_p = multiprocessing.Process(target=get_results, args=(q,))
    plot_results_p = multiprocessing.Process(target=plot_results, args=(q,))

    get_results_p.start()
    plot_results_p.start()

        # ax.scatter(track_plot.xs, track_plot.ys)
        #
        # fig.canvas.draw()
        # fig.canvas.flush_events()


def get_results(q):
    generator = track_objects_realtime()
    for item in generator:
        q.put(item)

        # print(item[2])


def plot_results(q):
    origin = [0, 0]

    track_ids = []
    track_plots = []

    plot_history = 200
    colours = [''] * plot_history
    for i in range(plot_history):
        colours[i] = scalar_to_rgb(i, plot_history)

    frame_no = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    while True:
        while not q.empty():
            item = q.get()
            tracks, (dx, dy), frame_no, frame = item[0], item[1], item[2], item[3]
            origin[0] += dx
            origin[1] += dy
            for track in tracks:
                track_id = track[0]
                if track_id not in track_ids:  # First occurrence of the track
                    track_ids.append(track_id)
                    track_plots.append(TrackPlot(track_id))

                track_plot = track_plots[track_ids.index(track_id)]
                track_plot.update(origin, track[3], frame_no)

        for track_plot in track_plots:
            idxs = np.where(np.logical_and(track_plot.frameNos > frame_no-plot_history,
                                           track_plot.frameNos <= frame_no))[0]
            print(f"{track_plot.id}: idx {idxs}")
            for idx in idxs:
                cv2.circle(frame, (track_plot.xs[idx], track_plot.ys[idx]),
                           3, colours[track_plot.frameNos[idx]-frame_no+plot_history-1][::-1], -1)
            if not len(idxs) == 0:
                cv2.putText(frame, str(track_plot.id), (track_plot.xs[idx], track_plot.ys[idx]),
                            font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)

        try:
            imshow_resized("plot", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass

    cv2.destroyAllWindows()


def delete_track_plots(frame_no):
    max_unseen = 1000


def scalar_to_hex(scalar_value, max_value):
    f = scalar_value / max_value
    a = (1-f)*5
    x = math.floor(a)
    y = math.floor(255*(a-x))
    if x == 0:
        return '#%02x%02x%02x' % (255, y, 0)
    elif x == 1:
        return '#%02x%02x%02x' % (255, 255, 0)
    elif x == 2:
        return '#%02x%02x%02x' % (0, 255, y)
    elif x == 3:
        return '#%02x%02x%02x' % (0, 255, 255)
    elif x == 4:
        return '#%02x%02x%02x' % (y, 0, 255)
    else: # x == 5:
        return '#%02x%02x%02x' % (255, 0, 255)


def scalar_to_rgb(scalar_value, max_value):
    f = scalar_value / max_value
    a = (1-f)*5
    x = math.floor(a)
    y = math.floor(255*(a-x))
    if x == 0:
        return (255, y, 0)
    elif x == 1:
        return (255, 255, 0)
    elif x == 2:
        return (0, 255, y)
    elif x == 3:
        return (0, 255, 255)
    elif x == 4:
        return (y, 0, 255)
    else: # x == 5:
        return (255, 0, 255)

if __name__ == "__main__":
    plot_tracks_realtime()