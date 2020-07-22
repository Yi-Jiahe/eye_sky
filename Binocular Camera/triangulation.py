import csv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    FoV_x = 1600/1920
    FoV_y = 1600/1080

    K_L = np.array([[1280*FoV_x, 0, 1280/2],
                    [0, 720*FoV_y, 720/2],
                    [0, 0, 1]])

    # K_L = np.array([[1920*FoV_x, 0, 1920/2],
    #                 [0, 1080*FoV_y, 1080/2],
    #                 [0, 0, 1]])

    K_R = np.array([[1920*FoV_x, 0, 1920/2],
                    [0, 1080*FoV_y, 1080/2],
                    [0, 0, 1]])

    B = 3.4

    # [frame_no, x_L, y_L, x_R, y_R]
    position_3D_temp = []

    # Left Camera
    with open('Trim_1/data_out_left.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for frame_no, row in enumerate(reader):
            position_3D_temp.append([frame_no, None, None, None, None])

            tracks_in_frame = int((len(row) - 1) / 4)
            for i in range(tracks_in_frame):
                track_data = row[(i * 4) + 1:(i * 4) + 4 + 1]
                id = int(track_data[0])

                # Segment 0
                # if id == 0:  # Tello
                # if id == 1:
                # Segment 1
                # if id == 22:
                # if id == 18:  # Mavic
                # Trim 1
                # if id == 0:  # Tello
                if id == 28:  # Mavic
                    position_3D_temp[frame_no][1] = int(track_data[1])
                    position_3D_temp[frame_no][2] = int(track_data[2])

    print(len(position_3D_temp))

    # Right Camera
    with open('Trim_1/data_out_right.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for frame_no, row in enumerate(reader):
            if frame_no >= len(position_3D_temp):
                position_3D_temp.append([frame_no, None, None, None, None])
            tracks_in_frame = int((len(row) - 1) / 4)
            for i in range(tracks_in_frame):
                track_data = row[(i * 4) + 1:(i * 4) + 4 + 1]
                id = int(track_data[0])

                # Segment 0
                # if id == 42 or id == 43:  # Tello
                # if id == 41:
                # Segment 1
                # if id == 0:
                # if id == 68:  # Mavic
                # Trim 1
                # if id == 2:  # Tello
                if id == 40:  # Mavic
                    x = int(track_data[1])
                    position_3D_temp[frame_no][3] = int(track_data[1])
                    position_3D_temp[frame_no][4] = int(track_data[2])

    frames = []
    epsilon = 0

    position_3D = []

    for frame in position_3D_temp:
        frame_no = frame[0]
        position_3D.append([frame_no, 0, 0, 0])
        if None not in frame:  # All values present
            x_L, y_L, x_R, y_R = frame[1:]

            alpha_L = np.arctan2(x_L - K_L[0, 2], K_L[0, 0]) / np.pi * 180
            alpha_R = np.arctan2(x_R - K_R[0, 2], K_R[0, 0]) / np.pi * 180

            # gamma = epsilon + alpha_L - alpha_R
            # l = l_0 * np.sqrt(1 / (2 * (1 - np.cos(gamma / 180 * np.pi))))

            Z = B / (np.tan((alpha_L + epsilon/2) * (np.pi / 180)) - np.tan((alpha_R + -epsilon/2) * (np.pi / 180)))

            print(f"X from X_L: {Z * np.tan((alpha_L + epsilon/2) * (np.pi / 180)) - B/2},"
                  f"X from X_R:{Z * np.tan((alpha_R + -epsilon/2) * (np.pi / 180)) + B/2}")
            X = (Z * np.tan((alpha_L + epsilon/2) * (np.pi / 180)) - B/2
                 + Z * np.tan((alpha_R + -epsilon/2) * (np.pi / 180)) + B/2) / 2

            print(f"Y from L: {Z * -(y_L-K_L[1, 2])/K_L[1, 1]},"
                  f"Y from R: {Z * -(y_R-K_R[1, 2])/K_R[1, 1]}")
            Y = (Z * -(y_L-K_L[1, 2])/K_L[1, 1] + Z * -(y_R-K_R[1, 2])/K_R[1, 1]) / 2

            # Accounting for tilt

            tilt = 5 * np.pi / 180
            R = np.array([[1, 0, 0],
                          [0, np.cos(tilt), np.sin(tilt)],
                          [0, -np.sin(tilt), np.cos(tilt)]])

            [X, Y, Z] = np.matmul(R, np.array([X, Y, Z]))
            # Account for raised cameras
            Y += 1.2

            print(f"(X, Y, Z) = {X, Y ,Z}")

            position_3D[frame_no][1:] = [X, Y, Z]

    with open('3D_positions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in position_3D:
            if None not in row:
                writer.writerow(row)

    xs = []
    ys = []
    zs = []

    for position in position_3D:
        xs.append(position[1])
        ys = np.append(ys, position[2])
        zs.append(position[3])

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')

    ax_3d.set_title('')

    ax_3d.set_xlabel('X/m')
    ax_3d.set_ylabel('Z/m')
    ax_3d.set_zlabel('Y/m')

    ax_3d.scatter(xs, zs, ys)

    plt.show()
