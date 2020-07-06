import csv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    fx = 1589.4987958913048
    cx = 967.1070079968151

    l_0 = 1.5

    alphas = []

    position_3D_temp = []

    with open('data_out_sony.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for frame_no, row in enumerate(reader):
            alphas.append([None, None])
            position_3D_temp.append([frame_no, None, None, None, None, None])

            tracks_in_frame = int((len(row) - 1) / 4)
            for i in range(tracks_in_frame):
                track_data = row[(i * 4) + 1:(i * 4) + 4 + 1]
                id = int(track_data[0])

                if id == 0:
                    x = int(track_data[1])
                    alpha = np.arctan2(x-cx, fx) / np.pi * 180
                    alphas[frame_no][0] = alpha

                    position_3D_temp[frame_no][1] = int(track_data[1])
                    position_3D_temp[frame_no][2] = int(track_data[2])

    with open('data_out_phone.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for frame_no, row in enumerate(reader):
            tracks_in_frame = int((len(row) - 1) / 4)
            for i in range(tracks_in_frame):
                track_data = row[(i * 4) + 1:(i * 4) + 4 + 1]
                id = int(track_data[0])

                if id == 22:
                    x = int(track_data[1])
                    alpha = np.arctan2(x - cx, fx) / np.pi * 180
                    alphas[frame_no][1] = alpha

                    position_3D_temp[frame_no][3] = int(track_data[1])
                    position_3D_temp[frame_no][4] = int(track_data[2])

    start_angle = 5
    stop_angle = 21

    frames = []
    l_range = []
    for n in range(start_angle, stop_angle):
        l_range.append([])

    out_data = []

    for frame_no, alpha in enumerate(alphas):
        if None not in alpha:
            frames.append(frame_no)

            row = [frame_no]

            for n, epsilon in enumerate(range(start_angle, stop_angle)):
                gamma = epsilon - alpha[0] + alpha[1]
                l = l_0 * np.sqrt(1/(2*(1-np.cos(gamma/180*np.pi))))
                l_range[n].append(l)
                row.append(l)

                if epsilon == 10:
                    position_3D_temp[frame_no][5] = l

            out_data.append(row)

    with open('depth_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in out_data:
            writer.writerow(row)

    position_3D = []
    for frame in position_3D_temp:
        if None not in frame:
            x = (frame[1] + frame[3])/2
            y = (frame[2] + frame[4])/2
            z = frame[5] * 20
            position_3D.append([frame[0], x, y, z])

    with open('3D_positions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in position_3D:
            if None not in row:
                writer.writerow(row)

    fig, ax = plt.subplots()

    ax.set_xlabel('Frame')
    ax.set_ylabel('L/m')

    for n, epsilon in enumerate(range(start_angle, stop_angle)):
        ax.plot(frames, l_range[n], label=f"{epsilon}deg")

    ax.legend()

    xs = []
    ys = np.array([])
    zs = []

    for position in position_3D:
        xs.append(position[1])
        ys = np.append(ys, position[2])
        zs.append(position[3])

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')

    ax_3d.plot(xs, zs, -ys)

    plt.show()
