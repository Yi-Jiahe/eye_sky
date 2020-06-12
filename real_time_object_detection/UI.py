import tkinter as tk
import threading
import time

from PIL import Image, ImageTk
import cv2

from real_time_object_detection.ObjectTracker import objectTracker
import track_visualisation_rt

class UI:
    def __init__(self):
        # initialize the root window and image panel
        self.root = tk.Tk()
        self.panel = None
        self.root.wm_title("Object Tracker and Visualiser")

        self.brightness_override_btn = tk.Button(self.root, text='Manually control brightness',
                                                 command=self.toggle_brightness_override)
        self.brightness_override_btn.pack()
        self.brightness_adjust_scale = tk.Scale(self.root, from_=0, to=256, orient=tk.HORIZONTAL,
                                                command=self.adjust_brightness)
        self.brightness_adjust_scale.pack()

        self.object_tracker = objectTracker()

        # Thread to pull output video
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):
        """
        The mainloop thread of Tkinter
        Raises:
            RuntimeError: To get around a RunTime error that Tkinter throws due to threading.
        """
        try:
            self.object_tracker.start_tracking()

            while not self.stopEvent.is_set():
                frame = self.object_tracker.get_original_output()

                if frame is None:
                    continue

                # transfer the format from frame to image
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self._updateGUIImage(image)

        except RuntimeError:
            print("Oh no, I personally don't know whats wrong")

    def _updateGUIImage(self, image):
        image = ImageTk.PhotoImage(image)

        if self.panel is None:
            self.panel = tk.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left", padx=10, pady=10)
            # otherwise, simply update the panel
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def toggle_brightness_override(self):
        self.object_tracker.brightness_override = not self.object_tracker.brightness_override

    def adjust_brightness(self, scale_value):
        self.object_tracker.brightness_threshold = int(scale_value)

    def onClose(self):
        """
        set the stop event, cleanup the camera, and allow the rest of the quit process to continue
        """
        print("[INFO] closing...")
        self.object_tracker.stop_process = True
        self.stopEvent.set()
        self.root.quit()