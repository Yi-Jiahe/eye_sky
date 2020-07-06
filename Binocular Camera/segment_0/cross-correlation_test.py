import csv
import numpy as np
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


def get_track_index(camera_position, camera_attitude, id, tracks):
    for idx, track in enumerate(tracks):
        track_camera_pos, track_attitude, track_id = track.identity()
        if track_camera_pos == camera_position and track_attitude == camera_attitude and track_id == id:
            return idx
    return None


def read_data(camera_data, views):
    for view_idx, data in enumerate(camera_data):
        with open(data[0], 'r') as csvfile:
            camera_position = data[1]
            camera_attitude = data[2]

            views.append([camera_position, camera_attitude, []])

            reader = csv.reader(csvfile)
            for row in reader:
                tracks_in_frame = int((len(row)-1) / 4)
                for i in range(tracks_in_frame):
                    track_data = row[(i*4)+1:(i*4)+4+1]
                    id = int(track_data[0])

                    tracks = views[view_idx][2]
                    idx = get_track_index(id, tracks)
                    if idx is None:
                        track = Track2D(camera_position, camera_attitude, id)
                        tracks.append(track)
                    else:
                        track = tracks[idx]
                    track.update(row[0], track_data[1], track_data[2], track_data[3])


def plot_Xj(views):
    fig, axes = plt.subplots(len(views))

    if len(views) > 1:
        for idx, track in enumerate(views):
            axes[idx].plot(track.frames, track.Xjs)
            axes[idx].set_title(f"Camera pos: {track.camera_position}, "
                                 f"Pan: {track.camera_attitude[0]}, Tilt: {track.camera_attitude[1]}, "
                                 f"ID: {track.id}")
            axes[idx].set_ylabel('Xj')
            axes[idx].set_xlabel('Frame')
            axes[idx].set_yscale('log')
    else:
        track = views[0][2][0]
        axes.plot(track.frames, track.Xjs)
        axes.set_title(f"Camera pos: {track.camera_position}, "
                            f"Pan: {track.camera_attitude[0]}, Tilt: {track.camera_attitude[1]}, "
                            f"ID: {track.id}")
        axes.set_ylabel('Xj')
        axes.set_xlabel('Frame')
        axes.set_yscale('log')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    camera_data = [['data_out_phone.csv', (0, 0, 0), (0, 0)],
                   ['data_out_sony.csv', (1, 0, 0), (0, 0)]]

    views = []

    read_data(camera_data, views)

    plot_Xj(views)
