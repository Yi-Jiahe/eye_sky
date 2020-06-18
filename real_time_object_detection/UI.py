import tkinter as tk
import threading
import time

from PIL import Image, ImageTk
import cv2

from real_time_object_detection.ObjectTracker import ObjectTracker
import track_visualisation_rt

class UI:
    def __init__(self):
        # initialize the root window and image panel
        self.root = tk.Tk()
        self.root.wm_title("Object Tracker and Visualiser")

        self.filename = 0

        self.start_tracking_btn = tk.Button(self.root, text='Start Tracking', command=self.start_tracking)
        self.start_tracking_btn.pack()

        self.filename = 0

        self.object_tracker = ObjectTracker()

        self.object_tracking_thread = None

        # set a callback to handle when the window is closed
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def start_tracking(self):
        control_panel = tk.Toplevel(self.root)
        self.brightness_override_btn = tk.Button(control_panel, text='Manually control brightness',
                                                 command=self.toggle_brightness_override)
        self.brightness_override_btn.pack()
        self.brightness_adjust_scale = tk.Scale(control_panel, from_=0, to=256, orient=tk.HORIZONTAL,
                                                command=self.adjust_brightness)
        self.brightness_adjust_scale.pack()

        self.object_tracking_thread = threading.Thread(target=self.object_tracker.track_objects, args=(self.filename,))
        self.object_tracking_thread.start()
        self.object_tracking_thread.join()

        print(self.object_tracking_thread)
        # self.root.mainloop()


    def toggle_brightness_override(self):
        self.object_tracker.brightness_override = not self.object_tracker.brightness_override

    def adjust_brightness(self, scale_value):
        self.object_tracker.brightness_threshold = int(scale_value)

    def onClose(self):
        """
        set the stop event, cleanup the camera, and allow the rest of the quit process to continue
        """
        print("[INFO] closing...")
        self.root.quit()