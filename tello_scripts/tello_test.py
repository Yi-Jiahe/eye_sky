import easytello

import cv2
import numpy as np
from math import sin, cos, radians, fabs
import imutils
import time
import threading
import socket
import tkinter


def easy_tello_stream():
    # drone = easytello.tello.tello_scripts(debug=False)
    #
    # drone.streamon()
    #
    # drone_query = threading.Thread(target=easy_tello_query, args=(drone,))

    drone_move = threading.Thread(target=easy_tello_movement)

    # drone_query.start()
    drone_move.start()

    drone_move.join()


def easy_tello_query(drone):

    previous_frame = None

    while True:
        # battery = drone.get_battery()
        # speed = drone.get_speed()
        # time = drone.get_time()
        # height = drone.get_height()
        # temp = drone.get_temp()
        pitch, roll, yaw = drone.get_attitude()
        # baro = drone.get_baro()
        # acceleration = drone.get_acceleration()
        # tof = drone.get_tof()
        # wifi = drone.get_wifi()

        # print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        # m = np.array([[cos(radians(roll)), -sin(radians(roll)), 0],
        #               [sin(radians(roll)), cos(radians(roll)), 0]])

        frame = drone.get_frame()
        if frame is not None:
            rows, columns, _ = frame.shape

            # print(f"Roll: {roll}")
            # print(f"sin: {sin(radians(roll))}, cos: {cos(radians(roll))}")
            # print(f"Width: {int(rows*cos(radians(roll))+columns*sin(radians(roll)))}, "
            #       f"Height: {int(rows*sin(radians(roll))+columns*cos(radians(roll)))}")

            # rot_mat = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), -roll, 1)
            #
            # print(rot_mat)
            # displacement_top_left = rot_mat[:, 2].copy()
            # rot_mat[:,2] = rot_mat[:,2]+displacement_top_left
            # print(rot_mat)
            #
            # frame_rotated = cv2.warpAffine(frame, rot_mat,
            #                                (int(rows*fabs(sin(radians(roll)))+columns*fabs(cos(radians(roll)))),
            #                                 int(rows*fabs(cos(radians(roll)))+columns*fabs(sin(radians(roll))))))

            frame_rotated = imutils.rotate_bound(frame, roll)

            if previous_frame is not None:
                image1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                image_points1 = cv2.goodFeaturesToTrack(image1, maxCorners=300, qualityLevel=0.01, minDistance=10)

                image_points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, image_points1, None)

                idx = np.where(status == 1)[0]
                image_points1 = image_points1[idx]
                image_points2 = image_points2[idx]

                # Estimate affine transformation
                # Produces a transformation matrix :
                # [[cos(theta).s, -sin(theta).s, tx],
                # [sin(theta).s, cos(theta).s, ty]]
                # where theta is rotation, s is scaling and tx,ty are translation
                m = cv2.estimateAffinePartial2D(image_points1, image_points2)[0]

                # Extract translation
                dx = m[0, 2]
                dy = m[1, 2]
                # print(f"dx:{dx}, dy=:{dy}")

            previous_frame = frame

            cv2.imshow('frame', frame_rotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    drone.streamoff()


def easy_tello_movement():
    pass


if __name__ == '__main__':
    easy_tello_stream()