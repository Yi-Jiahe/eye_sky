from numpy import sin, cos

import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools

import tkinter as tk


class UI:
    def __init__(self, cameras):
        self.cameras = cameras
        self.active_camera_no = 0
        self.active_camera = self.cameras[self.active_camera_no]

        self.root = tk.Tk()
        self.root.bind('<Key>', self.control)

        self.instructions = tk.Label(self.root, text="Click me to control the cameras \n\n"
                                                     "W, A, S, D to move, \n"
                                                     "Up, Down, Left, Right for pitch and tilt, \n"
                                                     "I, O to Zoom, \n"
                                                     "Tab to switch camera")
        self.instructions.pack()

        self.display = tk.Frame(self.root)
        self.display.pack()
        self.position_label = tk.Label(self.display, text=f"Position: {self.active_camera.position}")
        self.position_label.pack()
        self.f_label = tk.Label(self.display, text=f"Focal Length: {self.active_camera.f}")
        self.f_label.pack()



    def control(self, key):
        if key.keysym == 'Up':
            self.active_camera.psi += 0.1
        if key.keysym == 'Down':
            self.active_camera.psi -= 0.1
        if key.keysym == 'Left':
            self.active_camera.theta += 0.1
        if key.keysym == 'Right':
            self.active_camera.theta -= 0.1

        if key.keysym == 'w':
            self.active_camera.position[2] += 1
        if key.keysym == 's':
            self.active_camera.position[2] -= 1
        if key.keysym == 'a':
            self.active_camera.position[0] -= 1
        if key.keysym == 'd':
            self.active_camera.position[0] += 1
        if key.keysym == 'q':
            self.active_camera.position[1] += 1
        if key.keysym == 'e':
            self.active_camera.position[1] -= 1

        if key.keysym == 'i':
            self.active_camera.f += 0.1
        if key.keysym == 'o':
            self.active_camera.f -= 0.1

        if key.keysym == 'Tab':
            self.switch_active_camera()

        self.active_camera.update_projection_matrix()

        self.position_label['text'] = f"Position: {self.active_camera.position}"
        self.f_label['text'] = f"Focal Length: {self.active_camera.f}"

    def switch_active_camera(self):
        self.active_camera_no += 1
        if self.active_camera_no >= len(self.cameras):
            self.active_camera_no = 0
        self.active_camera = self.cameras[self.active_camera_no]


class Camera:
    def __init__(self, focal_length=1., resolution=(640, 480), rotation=(0., 0., 0.), position=(0., 0., 0.)):
        self.f = focal_length
        self.resolution = np.array(resolution)
        self.position = np.array(position)
        self.psi, self.theta, self.phi = rotation

        self.f_x, self.f_y = None, None
        self.c_x, self.c_y = None, None
        self.K = None
        self.R_c, self.C = None, None
        self.E = None
        self.P = None

        self.update_projection_matrix()

        self.detections = []

    def update_projection_matrix(self):
        self.f_x = -self.f*self.resolution[0]
        self.f_y = -self.f*self.resolution[1]
        self.c_x = self.resolution[0]/2
        self.c_y = self.resolution[1]/2
        self.K = np.array([[self.f_x, 0, self.c_x],
                           [0, -self.f_y, self.c_y],
                           [0, 0, 1]])
        R_roll = np.array([[1, 0, 0],
                           [0, cos(self.psi), sin(self.psi)],
                           [0, -sin(self.psi), cos(self.psi)]])
        R_pitch = np.array([[cos(self.theta), 0, -sin(self.theta)],
                            [0, 1, 0],
                            [sin(self.theta), 0, cos(self.theta)]])
        R_yaw = np.array([[cos(self.phi), sin(self.phi), 0],
                          [-sin(self.phi), cos(self.phi), 0],
                          [0, 0, 1]])
        R = R_roll @ R_pitch @ R_yaw
        self.R_c = R.T
        self.C = np.array(self.position)
        E_inv = np.zeros((4, 4))
        E_inv[3, 3] = 1
        E_inv[:3, :3] = self.R_c
        E_inv[:3, 3] = self.C.T
        self.E = np.linalg.inv(E_inv)

        self.P = self.K @ np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]]) @ self.E

    def camera_to_world(self, point_3D):
        return np.linalg.inv(self.E) @ np.append(point_3D, 1)

    def world_to_projection(self, point_3D):
        x, y, w = self.P @ np.append(point_3D, 1)
        # Sets projection to appear at infinity if object is behind the camera
        if w > 0: w = 0
        return x/w, y/w

    def projection_to_line(self, point_2D):
        ws = np.linspace(self.f, -20, 2)
        Xs, Ys, Zs = [], [], []
        for w in ws:
            x, y = np.array(point_2D)*w
            X, Y, Z = np.linalg.inv(self.K) @ np.array((x, y, w))
            X, Y, Z, _ = np.linalg.inv(self.E) @ np.append([X, Y, Z], 1)
            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)
        return Xs, Ys, Zs

    def image_plane(self):
        corners = ((0, 0),
                   (self.resolution[0], 0),
                   self.resolution,
                   (0, self.resolution[1]),
                   (0, 0))
        Xs, Ys, Zs = [], [], []
        for corner in corners:
            x, y = np.array(corner)*self.f
            X, Y, Z = np.linalg.inv(self.K) @ np.append((x, y), self.f)
            X, Y, Z, _ = np.linalg.inv(self.E) @ np.append([X, Y, Z], 1)
            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)
        return Xs, Ys, Zs


if __name__ == '__main__':
    object_colours = ('r', 'g', 'b')

    cameras = (Camera(focal_length=1., resolution=(640, 480), rotation=(0, 0, 0), position=(3, 1, 0)),
               Camera(focal_length=.7, resolution=(640, 480), rotation=(0, 0, 0), position=(0, 1, 0)))

    ui = UI(cameras)

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')

    ax_3d.set_xlim(-2, 6)
    ax_3d.set_ylim(0, 10)
    ax_3d.set_zlim(-20, 0)

    ax_3d.set_title('3D scene')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')

    ax_3d.view_init(90, -90)

    camera_plots = []
    for i, camera in enumerate(cameras):

        camera_location = ax_3d.text(camera.C[0] + 0.1, camera.C[1] + 0.1, camera.C[2], i)

        Xs, Ys, Zs = camera.projection_to_line((0, 0))
        top_left, = ax_3d.plot(Xs, Ys, Zs, color='k')
        Xs, Ys, Zs = camera.projection_to_line((camera.resolution[0], 0))
        top_right, = ax_3d.plot(Xs, Ys, Zs, color='k')
        Xs, Ys, Zs = camera.projection_to_line((0, camera.resolution[1]))
        bottom_left, = ax_3d.plot(Xs, Ys, Zs, color='k')
        Xs, Ys, Zs = camera.projection_to_line((camera.resolution[0], camera.resolution[1]))
        bottom_right, = ax_3d.plot(Xs, Ys, Zs, color='k')
        Xs, Ys, Zs = camera.image_plane()
        image_plane, = ax_3d.plot(Xs, Ys, Zs, color='k')

        camera_plots.append([camera_location, top_left, top_right, bottom_left, bottom_right, image_plane])

    objects = []
    for i in range(2):
        object = (np.random.random()*2, abs(np.random.random()*10), -10)
        print(object)
        objects.append(object)

        ax_3d.scatter(object[0], object[1], object[2], color=object_colours[i])

    camera = cameras[ui.active_camera_no]

    fig, ax = plt.subplots()
    ax.set_xlim(0, camera.resolution[0])
    ax.set_ylim(camera.resolution[1], 0)

    ax.set_title(f"Image from camera {ui.active_camera_no}")
    ax.set_xlabel('x/px')
    ax.set_ylabel('y/px')
    ax.set_xticks(np.arange(0, camera.resolution[0], 100))
    ax.set_yticks(np.arange(0, camera.resolution[1], 100))
    ax.grid(True)

    objects_2D = []
    for i, object in enumerate(objects):
        projection = camera.world_to_projection(object)
        object_2D = ax.scatter(projection[0], projection[1], color=object_colours[i])
        objects_2D.append(object_2D)

    plt.show(block=False)

    while True:
        camera = cameras[ui.active_camera_no]

        camera_location, top_left, top_right, bottom_left, bottom_right, image_plane =\
            camera_plots[ui.active_camera_no]

        camera_location._position3d = (camera.C[0] + 0.1, camera.C[1] + 0.1, camera.C[2])

        Xs, Ys, Zs = camera.projection_to_line((0, 0))
        top_left.set_data(Xs, Ys)
        top_left.set_3d_properties(Zs)
        Xs, Ys, Zs = camera.projection_to_line((camera.resolution[0], 0))
        top_right.set_data(Xs, Ys)
        top_right.set_3d_properties(Zs)
        Xs, Ys, Zs = camera.projection_to_line((0, camera.resolution[1]))
        bottom_left.set_data(Xs, Ys)
        bottom_left.set_3d_properties(Zs)
        Xs, Ys, Zs = camera.projection_to_line((camera.resolution[0], camera.resolution[1]))
        bottom_right.set_data(Xs, Ys)
        bottom_right.set_3d_properties(Zs)
        Xs, Ys, Zs = camera.image_plane()
        image_plane.set_data(Xs, Ys)
        image_plane.set_3d_properties(Zs)

        ax.set_title(f"Image from camera {ui.active_camera_no}")
        ax.set_xlim(0, camera.resolution[0])
        ax.set_ylim(camera.resolution[1], 0)
        ax.set_xticks(np.arange(0, camera.resolution[0], 100))
        ax.set_yticks(np.arange(0, camera.resolution[1], 100))

        detections = []
        for index, object in enumerate(objects):
            object_2D = objects_2D[index]
            projection = camera.world_to_projection(object)
            object_2D.set_offsets((projection[0], projection[1]))

            if 0 <= projection[0] <= camera.resolution[0] and 0 <= projection[1] <= camera.resolution[1]:
                detections.append(projection)
        camera.detections = detections

        fig_3d.canvas.draw()
        fig_3d.canvas.flush_events()
        fig.canvas.draw()
        fig.canvas.flush_events()

        print("#############################################")
        print('Ground Truth')
        for object in objects:
            print(object)

        print("Detections")

        for camera0, camera1 in itertools.combinations(cameras, 2):
            A = camera1.P @ np.append(camera0.position, 1)
            C = np.array([[0, -A[2], A[1]],
                          [A[2], 0, -A[0]],
                          [-A[1], A[0], 0]])
            F = C @ camera1.P @ np.linalg.pinv(camera0.P)

            for detection0 in camera0.detections:
                for detection1 in camera1.detections:

                    something = np.append(detection1, 1) @ F @ np.append(detection0, 1)
                    if np.abs(something) < 0.0001:
                        # print(f"Matched: {something}")

                        point_3D = cv2.triangulatePoints(camera0.P, camera1.P, detection0, detection1)

                        W = point_3D[3][0]
                        X, Y, Z = point_3D[0][0]/W, point_3D[1][0]/W, point_3D[2][0]/W

                        print(f"X:{X:.2f}, Y:{Y:.2f}, Z:{Z:.2f}")
                    # else:
                    #     print(f"No Match: {something}")