import tkinter as tk
import threading
from PIL import Image, ImageTk

import object_tracking_rt
import track_visualisation_rt

class UI:
    def __init__(self):
        self.thread = None  # thread of the Tkinter mainloop
        self.stopEvent = None

        # initialize the root window and image panel
        self.root = tk.Tk()
        self.panel = None

        # Thread to pull output video
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop(), args=())
        self.thread.start()

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


    def onClose(self):
        """
        set the stop event, cleanup the camera, and allow the rest of the quit process to continue
        """
        print("[INFO] closing...")
        self.stopEvent.set()
        self.root.quit()