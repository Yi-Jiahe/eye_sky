import csv
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


class Track2D:
    def __init__(self, camera_position, camera_attitude, id):
        self.camera_position = camera_position
        self.camera_attitude = camera_attitude
        self.id = id
        self.frames = []
        self.xs = np.array([])
        self.ys = np.array([])
        self.Xjs = []

    def identity(self):
        return self.camera_position, self.camera_attitude, self.id

    def update(self, frame, x, y, Xj):
        self.frames.append(int(frame))
        self.xs = np.append(self.xs, int(x))
        self.ys = np.append(self.ys, int(y))
        self.Xjs.append(float(Xj))


class Track3D:
    def __init__(self):
        pass


def get_track_index(camera_position, camera_attitude, id, tracks):
    for idx, track in enumerate(tracks):
        track_camera_pos, track_attitude, track_id = track.identity()
        if track_camera_pos == camera_position and track_attitude == camera_attitude and track_id == id:
            return idx
    return None


def read_data(camera_data, tracks):
    for data in camera_data:
        with open(data[0], 'r') as csvfile:
            camera_position = data[1]
            camera_attitude = data[2]
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 5:
                    # Need to split the row if there are multiple tracks
                    # But I'll ignore it for now
                    id = int(row[1])
                    idx = get_track_index(camera_position, camera_attitude, id, tracks)
                    if idx is None:
                        track = Track2D(camera_position, camera_attitude, id)
                        tracks.append(track)
                    else:
                        track = tracks[idx]
                    track.update(row[0], row[2], row[3], row[4])


def plot_Xj(tracks):
    fig, axes = plt.subplots(len(tracks))

    if len(tracks) > 1:
        for idx, track in enumerate(tracks):
            axes[idx].plot(track.frames, track.Xjs)
            axes[idx].set_title(f"Camera pos: {track.camera_position}, "
                                 f"Pan: {track.camera_attitude[0]}, Tilt: {track.camera_attitude[1]}, "
                                 f"ID: {track.id}")
            axes[idx].set_ylabel('Xj')
            axes[idx].set_xlabel('Frame')
            axes[idx].set_yscale('log')
    else:
        track = tracks[0]
        axes.plot(track.frames, track.Xjs)
        axes.set_title(f"Camera pos: {track.camera_position}, "
                            f"Pan: {track.camera_attitude[0]}, Tilt: {track.camera_attitude[1]}, "
                            f"ID: {track.id}")
        axes.set_ylabel('Xj')
        axes.set_xlabel('Frame')
        axes.set_yscale('log')

    plt.tight_layout()
    plt.show()


def plot_tracks_2D(tracks):
    for track in tracks:
        fig, ax = plt.subplots()

        ax.set_title(f"Camera pos: {track.camera_position}, "
                            f"Pan: {track.camera_attitude[0]}, Yaw: {track.camera_attitude[1]}, "
                            f"ID: {track.id}")
        ax.set_ylabel('-y-pos (camera frame)')
        ax.set_xlabel('x-pos (camera frame)')

        ax.plot(track.xs, -track.ys)

    plt.show()


if __name__ == '__main__':
    camera_data = [['data_out_side10763.csv', (0, 0, 0), (0, 0)],
                   ['data_out_top2.csv', (0, 1080, -360), (90, 90)]]

    tracks = []

    read_data(camera_data, tracks)

    Xj_1 = np.array(tracks[0].Xjs[::2])
    Xj_2 = np.array(tracks[1].Xjs)

    X = signal.correlate(Xj_1, Xj_2)

    fig, ax = plt.subplots()

    # plot_Xj(tracks)
    #
    # plot_tracks_2D(tracks)
    #
    # xs = tracks[0].xs
    # ys = -tracks[0].ys
    # zs = np.zeros(len(xs))
    # zs[::2] = tracks[1].xs[:len(zs)//2+1]
    # zs[1::2] = np.interp(np.arange(1,len(zs), 2), np.arange(0, len(zs),2), zs[::2])
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(xs, zs, ys)
    #
    # plt.show()