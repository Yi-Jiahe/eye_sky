import easytello

import cv2
import numpy as np
from math import sin, cos, radians
import time
import threading
import socket


def easy_tello_stream():
    drone = easytello.tello.Tello(debug=False)

    drone.streamon()

    drone_query = threading.Thread(target=easy_tello_query, args=(drone,))

    drone_query.start()

    drone_query.join()


def easy_tello_query(drone):
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

        print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

        frame = drone.get_frame()

        # m = np.array([[cos(radians(roll)), -sin(radians(roll)), 0],
        #               [sin(radians(roll)), cos(radians(roll)), 0]])

        if frame is not None:
            rows, columns, _ = frame.shape

            rot_mat = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), -roll, 1)
            print(rot_mat)
            frame_rotated = cv2.warpAffine(frame, rot_mat, (columns, rows))

            cv2.imshow('frame', frame_rotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    drone.streamoff()


if __name__ == '__main__':
    easy_tello_stream()