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
        self.times = np.array([])
        self.colourized_times = []
        self.lastSeen = 0

        self.Q = 0
        self.turning_angle = np.array([])  # Radians
        self.curvature = np.array([])
        self.pace = np.array([])
        self.track_feature_variable = 0


    def plot_track(self):
        print(f"Track {self.id} being plotted...")
        plt.scatter(self.xs, self.ys, c=self.colourized_times, marker='+')
        plt.show()

    def update(self, location, frame_no, time=None):
        self.xs = np.append(self.xs, [int(location[0])])
        self.ys = np.append(self.ys, [int(location[1])])
        self.frameNos = np.append(self.frameNos, [frame_no])

        if time is not None:
            self.times = np.append(self.times, time)

        self.lastSeen = frame_no

    def calculate_track_feature_angle(self):
        # Check if there is enough data to calculate turning angle (at least 3 points)
        if len(self.frameNos) >= 3:
            # Check if the last 3 frames are consecutive
            if self.frameNos[-2] == self.frameNos[-1] - 1 and self.frameNos[-3] == self.frameNos[-2] - 1:
                # retrieve the x and y values of the last 3 points
                x, y = self.xs[-3:], self.ys[-3:]
                t = 2  # introduce t for readability

                # Turning angle
                heading_1 = np.abs(np.arctan((y[t]   - y[t-1]) / (x[t]   - x[t-1])))
                heading_2 = np.abs(np.arctan((y[t-1] - y[t-2]) / (x[t-1] - x[t-2])))

                # if the denominator is 0
                heading_1 = 0 if np.isnan(heading_1) else heading_1
                heading_2 = 0 if np.isnan(heading_2) else heading_2

                turning_angle = np.abs( heading_2 - heading_1 )
                self.turning_angle = np.append(self.turning_angle, turning_angle)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ How to check for which quadrant it is in?

                # Curvature
                a = np.sqrt( (x[t]   - x[t-2])**2 + (y[t]   - y[t-2])**2 )
                b = np.sqrt( (x[t-1] - x[t-2])**2 + (y[t-1] - y[t-2])**2 )
                c = np.sqrt( (x[t]   - x[t-1])**2 + (y[t]   - y[t-1])**2 )

                curvature = np.arccos( (a**2 - b**2 - c**2) / 2*b*c )
                curvature = 0 if np.isnan(curvature) else curvature
                self.curvature = np.append(self.curvature, curvature)

                # Pace
                if self.times.size != 0:
                    dt = self.times[-1] - self.times[-2]
                else:
                    # assume 30 FPS
                    dt = 1/30

                pace = c/dt
                self.pace = np.append(self.pace, pace)

                self.track_feature_variable = np.mean(self.turning_angle) * np.mean(self.curvature) * np.mean(self.pace)



def plot_tracks_realtime(filename=0):
    plotting_queue = multiprocessing.Queue()

    get_results_p = multiprocessing.Process(target=get_results, args=(plotting_queue, filename))
    plot_results_p = multiprocessing.Process(target=plot_results, args=(plotting_queue, filename))

    get_results_p.start()
    plot_results_p.start()

    get_results_p.join()
    plot_results_p.join()


def get_results(q1, filename):
    generator = track_objects_realtime(filename)
    for item in generator:
        q1.put(item)


def plot_results(q, filename):
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

    cap = cv2.VideoCapture(filename)
    plot_out = cv2.VideoWriter('video_plot.mp4', cv2.VideoWriter_fourcc(*'h264'),
                               int(cap.get(cv2.CAP_PROP_FPS)),
                               (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    cap.release()

    new_data = False
    last_update = time.time()

    while True:
        while not q.empty():
            new_data = True

            item = q.get()
            tracks, origin, frame_no, frame, frame_start = item

            for track in tracks:
                track_id = track[0]
                if track_id not in track_ids:  # First occurrence of the track
                    track_ids.append(track_id)
                    track_plots.append(TrackPlot(track_id))

                track_plot = track_plots[track_ids.index(track_id)]

                if frame_start != False:
                    track_plot.update(track[3], frame_no, frame_start)
                else:
                    track_plot.update(track[3], frame_no)

            last_update = time.time()

        if new_data:
            for track_plot in track_plots:
                idxs = np.where(np.logical_and(track_plot.frameNos > frame_no-plot_history,
                                               track_plot.frameNos <= frame_no))[0]
                for idx in idxs:
                    cv2.circle(frame, (track_plot.xs[idx]-origin[0], track_plot.ys[idx]-origin[1]),
                               3, colours[track_plot.frameNos[idx]-frame_no+plot_history-1][::-1], -1)
                if len(idxs) != 0:
                    cv2.putText(frame, str(track_plot.id), (track_plot.xs[idx]-origin[0], track_plot.ys[idx]-origin[1]),
                                font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
                    track_plot.calculate_track_feature_angle()
                    cv2.putText(frame, str(track_plot.track_feature_variable),
                                (track_plot.xs[idx]-origin[0], track_plot.ys[idx]-origin[1] + 5),
                                font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
            plot_out.write(frame)
            imshow_resized("plot", frame)
            new_data = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if (time.time() - last_update) > 5 and not new_data:
            print("Timeout: Terminating plot")
            break

    plot_out.release()
    cv2.destroyAllWindows()


def delete_track_plots():
    pass


def rgb_to_hex(r,g,b):
    return '#%02x%02x%02x' % (r, g, b)


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
    plot_tracks_realtime('tiny_drones.mp4')