import cv2
import numpy as np
import threading
import multiprocessing
import time
import math

from filterpy.kalman import KalmanFilter
from automatic_brightness import average_brightness
from object_tracker import imshow_resized
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


class ObjectTracker:
    def __init__(self, filename=0):
        # Set-up video input stream
        self.filename=filename
        if self.filename == 0:
            self.real_time = True
        cap = cv2.VideoCapture(self.filename)
        self.cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.cap_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap_scaling = math.sqrt(self.cap_frame_width**2+self.cap_frame_height**2)/math.sqrt(848**2 + 480**2)
        cap.release()

        # Declare video output objects
        self.recording = False
        self.save_output = False

        self.debug = False
        # Frames for output
        self.frame_original = None
        self.frame_background_subtracted = None
        self.frame_to_be_removed = None
        self.frame_masked = None
        self.frame_out = None

        # Global reference frame
        self.origin = [0, 0]

        self.next_id = 0
        self.tracks = []

        self.frame_count = 0

        self.fps_log = []
        self.fps = self.cap_fps

        self.stop_process = True
        # Process for tracking loop
        self.tracking_loop_process = None
        self.tracking_loop_thread = None

        # For manual brightness adjustment
        self.brightness_override = False
        self.brightness_threshold = 127
        # For ground removal tuning
        self.ground_removal_override = False

        self.good_tracks = []

    def setup_system_objects(self):
        # varThreshold affects the spottiness of the image. The lower it is, the more smaller spots.
        # The larger it is, these spots will combine into large foreground areas
        fgbg = cv2.createBackgroundSubtractorMOG2(history=int(10 * self.cap_fps), varThreshold=64*self.cap_scaling,
                                                  detectShadows=False)
        # Background ratio represents the fraction of the history a frame must be present
        # to be considered part of the background
        # eg. history is 5s, background ratio is 0.1, frames present for 0.5s will be considered background
        fgbg.setBackgroundRatio(0.05)
        fgbg.setNMixtures(5)

        params = cv2.SimpleBlobDetector_Params()
        # params.filterByArea = True
        # params.minArea = 1
        # params.maxArea = 1000
        params.filterByConvexity = False
        params.filterByCircularity = False
        detector = cv2.SimpleBlobDetector_create(params)

        return fgbg, detector

    def track_objects(self):
        self.stop_process = False

        cap = cv2.VideoCapture(self.filename)

        videoWriter = None
        if self.recording:
            videoWriter = cv2.VideoWriter()

        combined_videoWriter = None
        if self.save_output:
            combined_videoWriter = cv2.VideoWriter

        fgbg, detector = self.setup_system_objects()

        frame_start = time.time()

        while cap.isOpened():
            frame_end = time.time()
            frame_time = frame_end - frame_start
            if frame_time > 0.001:
                self.fps_log.append(frame_time)
                if len(self.fps_log) > 5:
                    self.fps = 1 / (sum(self.fps_log) / len(self.fps_log))
                    self.fps_log.pop(0)

            ret, frame = cap.read()
            self.frame_original = frame.copy()
            # imshow_resized('original', frame)

            frame_start = time.time()

            if ret:
                # if downsample:
                #     frame = downsample_image(frame)
                # frame, mask = camera.undistort(frame)
                #
                # if self.frame_count == 0:
                #     frame_before = frame
                # elif self.frame_count >= 1:
                #     # Frame stabilization
                #     stabilized_frame, dx, dy = stabilize_frame(frame_before, frame)
                #     self.origin[0] += int(dx)
                #     self.origin[1] += int(dy)
                #
                #     frame_before = frame
                #     frame = stabilized_frame
                calibration_time = time.time()

                masked, centroids, sizes = self.detect_objects(frame, fgbg, detector)
                detection_time = time.time()

                self.predict_new_locations_of_tracks()
                prediction_time = time.time()

                assignments, unassigned_tracks, unassigned_detections\
                    = self.detection_to_track_assignment(centroids, 10*self.cap_scaling)
                assignment_time = time.time()

                self.update_assigned_tracks(assignments, centroids, sizes)

                self.update_unassigned_tracks(unassigned_tracks)
                self.delete_lost_tracks()
                self.create_new_tracks(unassigned_detections, centroids, sizes)

                masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
                self.good_tracks = self.filter_tracks(frame, masked)

                other_track_stuff = time.time()

                if self.recording:
                    self.videoWriter.write(self.frame_original)

                # if self.save_output:
                #     frame_out = np.zeros((FRAME_HEIGHT*2, FRAME_WIDTH, 3), dtype=np.uint8)
                #     frame_out[0:FRAME_HEIGHT, 0:FRAME_WIDTH] = frame
                #     frame_out[FRAME_HEIGHT:FRAME_HEIGHT*2, 0:FRAME_WIDTH] = masked
                #     out_combined.write(frame_out)

                display_time = time.time()

                print(f"The frame took {(display_time - frame_start)*1000}ms in total.\n"
                      f"Camera stabilization took {(calibration_time - frame_start)*1000}ms.\n"
                      f"Object detection took {(detection_time - calibration_time)*1000}ms.\n"
                      f"Prediction took {(prediction_time - detection_time)*1000}ms.\n"
                      f"Assignment took {(assignment_time - prediction_time)*1000}ms.\n"
                      f"Other track stuff took {(other_track_stuff - assignment_time)*1000}ms.\n"
                      f"Writing to file took {(display_time - other_track_stuff)*1000}ms.\n\n")

                self.frame_count += 1

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                if self.stop_tracking():
                    print('stopped')
                    break

            else:
                break

        cap.release()
        if videoWriter is not None:
            videoWriter.release()
        if combined_videoWriter is not None:
            combined_videoWriter.release()
        cv2.destroyAllWindows()

    # Apply image masks to prepare frame for blob detection
    # Masks: 1) Increased contrast and brightness to fade out the sky and make objects stand out
    #        2) Background subtractor to remove the stationary background (Converts frame to a binary image)
    #        3) Further background subtraction by means of contouring around non-circular objects
    #        4) Dilation to fill holes in detected drones
    #        5) Inversion to make the foreground black for the blob detector to identify foreground objects
    # Perform the blob detection on the masked image
    # Return detected blob centroids as well as size
    def detect_objects(self, frame, fgbg, detector, mask=None,):
        # Adjust contrast and brightness of image to make foreground stand out more
        # alpha used to adjust contrast, where alpha < 1 reduces contrast and alpha > 1 increases it
        # beta used to increase brightness, scale of (-255 to 255) ? Needs confirmation
        # formula is im_out = alpha * im_in + beta
        if not self.brightness_override:
            masked = cv2.convertScaleAbs(frame, alpha=1, beta=256-average_brightness(16, frame, mask)+15)
        else:
            masked = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness_threshold)
            print(self.brightness_threshold)

        # Subtract Background
        # Learning rate affects how often the model is updated
        # High values > 0.5 tend to lead to patchy output
        # Found that 0.1 - 0.3 is a good range
        masked = fgbg.apply(masked, learningRate=-1)

        if self.debug:
            self.frame_background_subtracted = masked.copy()

        masked = self.remove_ground(masked, int(13/(2.26/self.cap_scaling)), 0.7)

        # Morphological Transforms
        # Close to remove black spots
        # masked = imclose(masked, 3, 1)
        # Open to remove white holes
        # masked = imopen(masked, 3, 2)
        # masked = imfill(masked)
        kernel_dilation = np.ones((5, 5), np.uint8)
        masked = cv2.dilate(masked, kernel_dilation, iterations=2)

        # Invert frame such that black pixels are foreground
        masked = cv2.bitwise_not(masked)

        # keypoints = []
        # Blob detection
        keypoints = detector.detect(masked)

        n_keypoints = len(keypoints)
        centroids = np.zeros((n_keypoints, 2))
        sizes = np.zeros(n_keypoints)
        for i in range(n_keypoints):
            centroids[i] = keypoints[i].pt
            centroids[i] -= self.origin
            sizes[i] = keypoints[i].size

        return masked, centroids, sizes

    # Dilates the image multiple times to get of noise in order to get a single large contour for each background object
    # Identify background objects by their shape (non-circular)
    # Creates a copy of the input image which has the background contour filled in
    # Returns the filled image which has the background elements filled in
    def remove_ground(self, im_in, dilation_iterations, background_contour_circularity):
        kernel_dilation = np.ones((5, 5), np.uint8)
        # Number of iterations determines how close objects need to be to be considered background
        dilated = cv2.dilate(im_in, kernel_dilation, iterations=dilation_iterations)

        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        background_contours = []
        for contour in contours:
            # Identify background from foreground by the circularity of their dilated contours
            circularity = 4 * math.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
            if circularity <= background_contour_circularity:
                background_contours.append(contour)

        if self.debug:
            # This bit is used to find a suitable level of dilation to remove background objects
            # while keeping objects to be detected
            # im_debug = cv2.cvtColor(im_in.copy(), cv2.COLOR_GRAY2BGR)
            im_debug = self.frame_original.copy()
            cv2.drawContours(im_debug, background_contours, -1, (0, 255, 0), 3)

        im_out = im_in
        cv2.drawContours(im_out, background_contours, -1, 0, -1)

        return im_out

    def predict_new_locations_of_tracks(self):
        for track in self.tracks:
            track.kalmanFilter.predict()

    # Assigns detections to tracks using Munkre's Algorithm with cost based on euclidean distance,
    # with detections being located too far from existing tracks being designated as unassigned detections
    # and tracks without any nearby detections being designated as unassigned tracks
    def detection_to_track_assignment(self, centroids, cost_of_non_assignment):
        # start_time = time.time()
        m, n = len(self.tracks), len(centroids)
        k, l = min(m, n), max(m, n)

        # Create a square 2-D cost matrix with dimensions twice the size of the larger list (detections or tracks)
        cost = np.zeros((k + l, k + l))
        # initialization_time = time.time()

        # Calculate the distance of every detection from each track,
        # filling up the rows of the cost matrix (up to column n, the number of detections) corresponding to existing tracks
        # This creates a m x n matrix
        for i in range(len(self.tracks)):
            start_time_distance_loop = time.time()
            track = self.tracks[i]
            track_location = track.kalmanFilter.x[:2]
            cost[i, :n] = np.array([distance.euclidean(track_location, centroid) for centroid in centroids])
        # distance_time = time.time()

        unassigned_track_cost = cost_of_non_assignment
        unassigned_detection_cost = cost_of_non_assignment

        extra_tracks = 0
        extra_detections = 0
        if m > n:  # More tracks than detections
            extra_tracks = m - n
        elif n > m:  # More detections than tracks
            extra_detections = n - m
        elif n == m:
            pass

        # Padding cost matrix with dummy columns to account for unassigned tracks
        # This is used to fill the top right corner of the cost matrix
        detection_padding = np.ones((m, m)) * unassigned_track_cost
        cost[:m, n:] = detection_padding

        # Padding cost matrix with dummy rows to account for unassigned detections
        # This is used to fill the bottom left corner of the cost matrix
        track_padding = np.ones((n, n)) * unassigned_detection_cost
        cost[m:, :n] = track_padding
        # padding_time = time.time()

        # The bottom right corner of the cost matrix, corresponding to dummy detections being matched to dummy tracks
        # is left with 0 cost to ensure that excess dummies are always matched to each other

        # Perform the assignment, returning the indices of assignments,
        # which are combined into a coordinate within the cost matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        assignments_all = np.column_stack((row_ind, col_ind))
        # assignment_time = time.time()

        # Assignments within the top left corner corresponding to existing tracks and detections
        # are designated as (valid) assignments
        assignments = assignments_all[(assignments_all < [m, n]).all(axis=1)]
        # Assignments within the top right corner corresponding to existing tracks matched with dummy detections
        # are designated as unassigned tracks and will later be regarded as invisible
        unassigned_tracks = assignments_all[
            (assignments_all >= [0, n]).all(axis=1) & (assignments_all < [m, k + l]).all(axis=1)]
        # Assignments within the bottom left corner corresponding to detections matched to dummy tracks
        # are designated as unassigned detections and will generate a new track
        unassigned_detections = assignments_all[
            (assignments_all >= [m, 0]).all(axis=1) & (assignments_all < [k + l, n]).all(axis=1)]
        # sorting_time = time.time()

        # print(f"Initialization took {initialization_time - start_time}ms.\n"
        #       f"Distance measuring took {distance_time - initialization_time}ms.\n"
        #       f"Padding took {padding_time - distance_time}ms.\n"
        #       f"Assignment took {assignment_time - padding_time}ms.\n"
        #       f"Sorting took {sorting_time - assignment_time}\n\n")

        return assignments, unassigned_tracks, unassigned_detections

    # Using the coordinates of valid assignments which correspond to the detection and track indices,
    # update the track with the matched detection
    def update_assigned_tracks(self, assignments, centroids, sizes):
        for assignment in assignments:
            track_idx = assignment[0]
            detection_idx = assignment[1]
            centroid = centroids[detection_idx]
            size = sizes[detection_idx]

            track = self.tracks[track_idx]

            kf = track.kalmanFilter
            kf.update(centroid)

            # # Adaptive filtering
            # # If the residual is too large, increase the process noise
            # Q_scale_factor = 100.
            # y, S = kf.y, kf.S  # Residual and Measurement covariance
            # # Square and normalize the residual
            # eps = np.dot(y.T, np.linalg.inv(S)).dot(y)
            # kf.Q *= eps * 10.

            track.size = size
            track.age += 1

            track.totalVisibleCount += 1
            track.consecutiveInvisibleCount = 0

    # Existing tracks without a matching detection are aged and considered invisible for the frame
    def update_unassigned_tracks(self, unassigned_tracks):
        for unassignedTrack in unassigned_tracks:
            track_idx = unassignedTrack[0]

            track = self.tracks[track_idx]

            track.age += 1
            track.consecutiveInvisibleCount += 1

    # If any track has been invisible for too long, or generated by a flash, it will be removed from the list of tracks
    def delete_lost_tracks(self):
        if len(self.tracks) == 0:
            return

        invisible_for_too_long = 3 * self.fps
        age_threshold = 1 * self.fps

        tracks_to_be_removed = []

        for track in self.tracks:
            visibility = track.totalVisibleCount / track.age
            # A new created track with a low visibility is likely to have been generated by noise and is to be removed
            # Tracks that have not been seen for too long (The threshold determined by the reliability of the filter)
            # cannot be accurately located and are also be removed
            if (track.age < age_threshold and visibility < 0.8) \
                    or track.consecutiveInvisibleCount >= invisible_for_too_long:
                tracks_to_be_removed.append(track)

        self.tracks = [track for track in self.tracks if track not in tracks_to_be_removed]

    # Detections not assigned an existing track are given their own track, initialized with the location of the detection
    def create_new_tracks(self, unassigned_detections, centroids, sizes):
        for unassignedDetection in unassigned_detections:
            detection_idx = unassignedDetection[1]
            centroid = centroids[detection_idx]
            size = sizes[detection_idx]

            dt = 1 / self.fps  # Time step between measurements in seconds

            track = Track(self.next_id, size)

            # Constant velocity model
            # Initial Location
            track.kalmanFilter.x = [centroid[0], centroid[1], 0, 0]
            # State Transition Matrix
            track.kalmanFilter.F = np.array([[1., 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]])
            # Measurement Function
            track.kalmanFilter.H = np.array([[1., 0, 0, 0],
                                             [0, 1, 0, 0]])
            # Ah I really don't know what I'm doing here
            # Covariance Matrix
            track.kalmanFilter.P = np.diag([200., 200, 50, 50])
            # Motion Noise
            track.kalmanFilter.Q = np.diag([100., 100, 25, 25])
            # Measurement Noise
            track.kalmanFilter.R = 100
            # # Constant acceleration model

            self.tracks.append(track)

            self.next_id += 1

    def filter_tracks(self, frame, masked):
        # Minimum number of frames to remove noise seems to be somewhere in the range of 30
        # Actually, I feel having both might be redundant together with the deletion criteria
        min_track_age = max(1.0*self.fps, 30)  # seconds * FPS to give number of frames in seconds
        # This has to be less than or equal to the minimum age or it make the minimum age redundant
        min_visible_count = max(1.0*self.fps, 30)

        good_tracks = []

        if len(self.tracks) != 0:
            for track in self.tracks:
                if track.age > min_track_age and track.totalVisibleCount > min_visible_count:
                    centroid = track.kalmanFilter.x[:2]
                    size = track.size

                    good_tracks.append([track.id, track.age, size, (centroid[0], centroid[1])])

                    centroid = track.kalmanFilter.x[:2] + self.origin

                    # Display filtered tracks
                    rect_top_left = (int(centroid[0] - size / 2), int(centroid[1] - size / 2))
                    rect_bottom_right = (int(centroid[0] + size / 2), int(centroid[1] + size / 2))
                    colour = (0, 0, 255)
                    thickness = 1
                    cv2.rectangle(frame, rect_top_left, rect_bottom_right, colour, thickness)
                    cv2.rectangle(masked, rect_top_left, rect_bottom_right, colour, thickness)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(frame, str(track.id), (rect_bottom_right[0], rect_top_left[1]),
                                font, font_scale, colour, thickness, cv2.LINE_AA)
                    cv2.putText(masked, str(track.id), (rect_bottom_right[0], rect_top_left[1]),
                                font, font_scale, colour, thickness, cv2.LINE_AA)
        self.frame_original = frame
        self.frame_masked = frame

        return good_tracks

    def start_tracking(self):
        # self.tracking_loop_process = multiprocessing.Process(target=self.track_objects, args=())
        # self.tracking_loop_process.start()
        self.tracking_loop_thread = threading.Thread(target=self.track_objects, args=())
        self.tracking_loop_thread.start()

    # frame retrieval methods
    def get_original_output(self):
        return self.frame_original

    def get_masked_output(self):
        return self.frame_masked

    def stop_tracking(self):
        return self.stop_process

class Track:
    def __init__(self, track_id, size):
        self.id = track_id
        self.size = size
        # Constant Velocity Model
        self.kalmanFilter = KalmanFilter(dim_x=4, dim_z=2)
        # # Constant Acceleration Model
        # self.kalmanFilter = KalmanFilter(dim_x=6, dim_z=2)
        self.age = 1
        self.totalVisibleCount = 1
        self.consecutiveInvisibleCount = 0