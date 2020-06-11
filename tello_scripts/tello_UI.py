import tkinter as tk
import threading
from PIL import Image, ImageTk
import time


class TelloUI:
    def __init__(self, tello):
        self.tello = tello # videostream device
        self.frame = None
        self.thread = None  # thread of the Tkinter mainloop
        self.stopEvent = None

        # control variables
        self.distance = 0.1  # default distance for 'move' cmd
        self.degree = 30  # default degree for 'cw' or 'ccw' cmd

        # initialize the root window and image panel
        self.root = tk.Tk()
        self.panel = None

        # binding arrow keys to drone control
        self.root.bind('<KeyPress-w>', self.on_keypress_w)
        self.root.bind('<KeyPress-a>', self.on_keypress_a)
        self.root.bind('<KeyPress-s>', self.on_keypress_s)
        self.root.bind('<KeyPress-d>', self.on_keypress_d)
        self.root.bind('<KeyPress-Up>', self.on_keypress_up)
        self.root.bind('<KeyPress-Down>', self.on_keypress_down)
        self.root.bind('<KeyPress-Left>', self.on_keypress_left)
        self.root.bind('<KeyPress-Right>', self.on_keypress_right)

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # # the sending_command will send command to tello every 5 seconds
        # self.sending_command_thread = threading.Thread(target = self._sendingCommand)

        # set a callback to handle when the window is closed
        self.root.wm_title("TELLO Controller")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):
        """
        The mainloop thread of Tkinter
        Raises:
            RuntimeError: To get around a RunTime error that Tkinter throws due to threading.
        """
        try:
            while not self.stopEvent.is_set():

                self.frame = self.tello.get_frame()

                if self.frame is None:
                    continue

                # transfer the format from frame to image
                image = Image.fromarray(self.frame)

                self._updateGUIImage(image)

            pass
        except RuntimeError:
            print("Oh no, I personally don't know whats wrong")

    def _updateGUIImage(self,image):
        image = ImageTk.PhotoImage(image)

        if self.panel is None:
            self.panel = tk.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left", padx=10, pady=10)
            # otherwise, simply update the panel
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    # def _sendingCommand(self):
    #     """
    #     start a while loop that sends 'command' to tello every 5 second
    #     """
    #     while True:
    #         self.tello.send_command('command')
    #         time.sleep(5)

    def telloTakeOff(self):
        return self.tello.takeoff()

    def telloLanding(self):
        return self.tello.land()

    def telloMoveForward(self, distance):
        return self.tello.move_forward(distance)

    def telloMoveBackward(self, distance):
        return self.tello.move_backward(distance)

    def telloMoveLeft(self, distance):
        return self.tello.move_left(distance)

    def telloMoveRight(self, distance):
        return self.tello.move_right(distance)

    def telloUp(self, dist):
        return self.tello.move_up(dist)

    def telloDown(self, dist):
        return self.tello.move_down(dist)

    def telloCW(self, degree):
        return self.tello.rotate_cw(degree)

    def telloCCW(self, degree):
        return self.tello.rotate_ccw(degree)

    def updateDistancebar(self):
        self.distance = self.distance_bar.get()

    def updateDegreebar(self):
        self.degree = self.degree_bar.get()

    def on_keypress_w(self, event):
        self.telloMoveForward(self.distance)

    def on_keypress_a(self, event):
        self.telloMoveLeft(self.distance)

    def on_keypress_s(self, event):
        self.telloMoveBackward(self.distance)

    def on_keypress_d(self, event):
        self.telloMoveRight(self.distance)

    def on_keypress_up(self, event):
        self.telloUp(self.distance)

    def on_keypress_down(self, event):
        self.telloDown(self.distance)

    def on_keypress_left(self, event):
        self.tello.rotate_ccw(self.degree)

    def on_keypress_right(self, event):
        self.tello.rotate_cw(self.degree)

    def onClose(self):
        """
        set the stop event, cleanup the camera, and allow the rest of

        the quit process to continue
        """
        print("[INFO] closing...")
        self.stopEvent.set()
        del self.tello
        self.root.quit()