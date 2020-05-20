import cv2
import math
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

import time


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


# ---------------------------------------Start Section------------------------------------------#
# Implementation of imopen from Matlab:
def imopen(im_in, kernel_size, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)/(kernel_size ^ 2)
    im_out = cv2.morphologyEx(im_in, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return im_out


# Implementation of imclose from Matlab:
def imclose(im_in, kernel_size, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)/(kernel_size ^ 2)
    im_out = cv2.morphologyEx(im_in, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return im_out


# Implementation of imfill() from Matlab
# Based off https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
# The idea is to add the inverse of the holes to fill the holes
def imfill(im_in):
    # Step 1: Threshold to obtain a binary image
    # Values above 220 to 0, below 220 to 255. (Inverse threshold)
    th, im_th = cv2.threshold(im_in, 220, 225, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image
    # im_floodfill = im_th.copy

    # Step 2: Floodfill the thresholded image
    # Mask used to flood fill
    # Note mask has to be 2 pixels larger than the input image
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    _, _, im_floodfill, _ = cv2.floodFill(im_th, mask, (0, 0), 255)

    # Step 3: Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Step 4: Combine the two images to get the foreground image with holes filled in
    # Floodfilled image needs to be trimmed to perform the bitwise or operation.
    # Trimming is done from the outside. I.e. the "Border" is removed
    im_out = cv2.bitwise_or(im_th, im_floodfill_inv[1:-1, 1:-1])

    return im_out
# ---------------------------------------End Section------------------------------------------#


def motion_based_multi_object_tracking(filename):
    cap = cv2.VideoCapture(filename)

    global FPS, FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    FRAME_WIDTH = int(cap.get(3))
    FRAME_HEIGHT = int(cap.get(4))
    # The scaling factor is the ratio of the diagonal of the input frame
    # to the video used to test the parameters, which in this case is 848x480
    SCALE_FACTOR = int(math.sqrt(FRAME_WIDTH ^ 2 + FRAME_HEIGHT ^ 2)/math.sqrt(848 ^ 2 + 480 ^ 2))

    out_original = cv2.VideoWriter('out_original.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                   FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    out_masked = cv2.VideoWriter('out_masked.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                 FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    fgbg, detector = setup_system_objects(filename)

    tracks = []

    next_id = 0
    frame_count = 0

    centroids_log = []
    tracks_log = []

    good_tracks_log = []
    good_tracks_log.append([FRAME_WIDTH, FRAME_HEIGHT])

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            start_time = time.time()

            # if frame_count == 0:
            #     frame_before = frame
            # elif frame_count >= 1:
            #     image1 = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
            #     image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #
            #     image_points1 = cv2.goodFeaturesToTrack(image1, maxCorners=100, qualityLevel=0.01, minDistance=10)
            #
            #     image_points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, image_points1, None)
            #
            #     idx = np.where(status == 1)[0]
            #     image_points1 = image_points1[idx]
            #     image_points2 = image_points2[idx]
            #
            #     m = cv2.estimateAffinePartial2D(image_points1, image_points2)[0]
            #     rows, columns = image1.shape
            #     frame_stabilized = cv2.warpAffine(frame, m, (columns, rows))
            #
            #     frame_before = frame
            #     frame = frame_stabilized

            calibration_time = time.time()

            centroids, sizes, masked = detect_objects(frame, fgbg, detector)
            detection_time = time.time()
            centroids_log.append(centroids)

            predict_new_locations_of_tracks(tracks)
            prediction_time = time.time()

            assignments, unassigned_tracks, unassigned_detections = detection_to_track_assignment(tracks, centroids)
            assignment_time = time.time()

            update_assigned_tracks(assignments, tracks, centroids, sizes)
            update_time = time.time()
            tracks_log.append(tracks)

            update_unassigned_tracks(unassigned_tracks, tracks)
            update_unassigned_time = time.time()
            tracks = delete_lost_tracks(tracks)
            deletion_time = time.time()
            next_id = create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes)
            creation_time = time.time()

            good_tracks = display_tracking_results(frame, masked, tracks, frame_count, out_original, out_masked)
            display_time = time.time()

            print(f"The frame took {(display_time - start_time)*1000}ms in total.\n"
                  f"Camera stabilization took {(calibration_time - start_time)*1000}ms.\n"
                  f"Object detection took {(detection_time - calibration_time)*1000}ms.\n"
                  f"Prediction took {(prediction_time - calibration_time)*1000}ms.\n"
                  f"Assignment took {(assignment_time - prediction_time)*1000}ms.\n"
                  f"Updating took {(update_time - assignment_time)*1000}ms.\n"
                  f"Updating unassigned tracks took {(update_unassigned_time - update_time)*1000}.\n"
                  f"Deletion took {(deletion_time - update_unassigned_time)*1000}ms.\n"
                  f"Track creation took {(creation_time - deletion_time)*1000}ms.\n"
                  f"Display took {(display_time - creation_time)*1000}ms.\n\n")

            if good_tracks:
                good_tracks_log.append(good_tracks)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    out_original.release()
    cv2.destroyAllWindows()

    return good_tracks_log


# Create VideoCapture object to extract frames from,
# background subtractor object and blob detector objects for object detection
# and VideoWriters for output videos
def setup_system_objects(filename):
    # varThreshold affects the spottiness of the image. The lower it is, the more smaller spots.
    # The larger it is, these spots will combine into large foreground areas
    fgbg = cv2.createBackgroundSubtractorMOG2(history=int(15*FPS), varThreshold=64 * SCALE_FACTOR, detectShadows=False)
    # Background ratio represents the fraction of the history a frame must be present
    # to be considered part of the background
    # eg. history is 5s, background ratio is 0.1, frames present for 0.5s will be considered background
    fgbg.setBackgroundRatio(0.05)
    fgbg.setNMixtures(5)

    params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.minArea = 1
    # params.maxArea = 1000
    detector = cv2.SimpleBlobDetector_create(params)

    return fgbg, detector


# Apply image masks to prepare frame for blob detection
# Masks: 1) Increased contrast and brightness to fade out the sky and make objects stand out
#        2) Background subtractor to remove the stationary background (Converts frame to a binary image)
#        3) Inversion to make the foreground black for the blob detector to identify foreground objects
#        4) Closing mask to remove black spots
# Perform the blob detection on the masked image
# Return detected blob centroids as well as size
def detect_objects(frame, fgbg, detector):
    # Adjust contrast and brightness of image to make foreground stand out more
    # alpha used to adjust contrast, where alpha < 1 reduces contrast and alpha > 1 increases it
    # beta used to increase brightness, scale of -255? to 255
    masked = cv2.convertScaleAbs(frame, alpha=1, beta=100)
    # masked = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

    # Subtract Background
    # Learning rate affects how aggressively the algorithm applies the changes to background ratio and stuff
    # Or so I believe. Adjust it alongside background ratio and history to tune
    masked = fgbg.apply(masked, learningRate=0.3)

    # Invert frame such that black pixels are foreground
    masked = cv2.bitwise_not(masked)

    # Close to remove black spots
    masked = imclose(masked, 3, 1)
    # Open to remove white holes
    # masked = imopen(masked, 3, 2)
    # masked = imfill(masked)

    # Blob detection
    keypoints = detector.detect(masked)

    n_keypoints = len(keypoints)
    centroids = np.zeros((n_keypoints, 2))
    sizes = np.zeros(n_keypoints)
    for i in range(n_keypoints):
        centroids[i] = keypoints[i].pt
        sizes[i] = keypoints[i].size

    return centroids, sizes, masked


def predict_new_locations_of_tracks(tracks):
    for track in tracks:
        track.kalmanFilter.predict()


# Assigns detections to tracks using Munkre's Algorithm with cost based on euclidean distance,
# with detections being located too far from existing tracks being designated as unassigned detections
# and tracks without any nearby detections being designated as unassigned tracks
def detection_to_track_assignment(tracks, centroids):
    # start_time = time.time()
    n, m = len(tracks), len(centroids)
    k, l = min(n, m), max(n, m)

    # Create a square 2-D cost matrix with dimensions twice the size of the larger list (detections or tracks)
    cost = np.zeros((l * 2, l * 2))
    # initialization_time = time.time()

    # Calculate the distance of every detection from each track,
    # filling up the rows of the cost matrix (up to column m, the number of detections) corresponding to existing tracks
    for i in range(len(tracks)):
        start_time_distance_loop = time.time()
        track = tracks[i]
        track_location = track.kalmanFilter.x[:2]
        cost[i, :m] = np.array([distance.euclidean(track_location, centroid) for centroid in centroids])
    # distance_time = time.time()

    cost_of_non_assignment = 20
    unassigned_track_cost = cost_of_non_assignment
    unassigned_detection_cost = cost_of_non_assignment

    extra_tracks = 0
    extra_detections = 0
    if n > m:  # More tracks than detections
        extra_tracks = n - m
    elif m > n:  # More detections than tracks
        extra_detections = m - n
    elif n == m:
        pass

    # Padding cost matrix with dummy columns to account for unassigned tracks
    # This is used to fill the top right corner of the cost matrix
    detection_padding = np.ones((l, l + extra_tracks)) * unassigned_track_cost
    cost[:l, m:] = detection_padding

    # Padding cost matrix with dummy rows to account for unassigned detections
    # This is used to fill the bottom left corner of the cost matrix
    track_padding = np.ones((l + extra_detections, l)) * unassigned_detection_cost
    cost[n:, :l] = track_padding
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
    assignments = assignments_all[(assignments_all < [n, m]).all(axis=1)]
    # Assignments within the top right corner corresponding to existing tracks matched with dummy detections
    # are designated as unassigned tracks and will later be regarded as invisible
    unassigned_tracks = assignments_all[
        (assignments_all >= [0, m]).all(axis=1) & (assignments_all < [l, l * 2]).all(axis=1)]
    # Assignments within the bottom left corner corresponding to detections matched to dummy tracks
    # are designated as unassigned detections and will generate a new track
    unassigned_detections = assignments_all[
        (assignments_all >= [n, 0]).all(axis=1) & (assignments_all < [l * 2, l]).all(axis=1)]
    # sorting_time = time.time()

    # print(f"Initialization took {initialization_time - start_time}ms.\n"
    #       f"Distance measuring took {distance_time - initialization_time}ms.\n"
    #       f"Padding took {padding_time - distance_time}ms.\n"
    #       f"Assignment took {assignment_time - padding_time}ms.\n"
    #       f"Sorting took {sorting_time - assignment_time}\n\n")

    return assignments, unassigned_tracks, unassigned_detections


# Using the coordinates of valid assignments which correspond to the detection and track indices,
# update the track with the matched detection
def update_assigned_tracks(assignments, tracks, centroids, sizes):
    for assignment in assignments:
        track_idx = assignment[0]
        detection_idx = assignment[1]
        centroid = centroids[detection_idx]
        size = sizes[detection_idx]

        track = tracks[track_idx]
        track.kalmanFilter.update(centroid)
        track.size = size
        track.age += 1

        track.totalVisibleCount += 1
        track.consecutiveInvisibleCount = 0


# Existing tracks without a matching detection are aged and considered invisible for the frame
def update_unassigned_tracks(unassigned_tracks, tracks):
    for unassignedTrack in unassigned_tracks:
        track_idx = unassignedTrack[0]

        track = tracks[track_idx]

        track.age += 1
        track.consecutiveInvisibleCount += 1


# If any track has been invisible for too long, or generated by a flash, it will be removed from the list of tracks
def delete_lost_tracks(tracks):
    if len(tracks) == 0:
        return tracks

    invisible_for_too_long = 3 * FPS
    age_threshold = 1 * FPS

    tracks_to_be_removed = []

    for track in tracks:
        visibility = track.totalVisibleCount/track.age
        # A new created track with a low visibility is likely to have been generated by noise and is to be removed
        # Tracks that have not been seen for too long (The threshold determined by the reliability of the filter)
        # cannot be accurately located and are also be removed
        if (track.age < age_threshold and visibility < 0.8) \
                or track.consecutiveInvisibleCount >= invisible_for_too_long:
            tracks_to_be_removed.append(track)

    tracks = [track for track in tracks if track not in tracks_to_be_removed]

    return tracks


# Detections not assigned an existing track are given their own track, initialized with the location of the detection
def create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes):
    for unassignedDetection in unassigned_detections:
        detection_idx = unassignedDetection[1]
        centroid = centroids[detection_idx]
        size = sizes[detection_idx]

        track = Track(next_id, size)
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
        # Covariance Matrix
        track.kalmanFilter.P = np.diag([200., 200, 50, 50])
        # Motion Noise
        track.kalmanFilter.Q = np.diag([100., 100, 25, 25])
        # Measurement Noise
        track.kalmanFilter.R = 100
        # # Constant acceleration model
        # # Initial Location
        # track.kalmanFilter.x = [centroid[0], centroid[1], 0, 0, 0, 0]
        # # State Transition Matrix
        # track.kalmanFilter.F = np.array([[1., 0, 1, 0, 0.5, 0],
        #                                  [0, 1, 0, 1, 0, 0.5],
        #                                  [0, 0, 1, 0, 1, 0],
        #                                  [0, 0, 0, 1, 0, 1],
        #                                  [0, 0, 0, 0, 1, 0],
        #                                  [0, 0, 0, 0, 0, 1]])
        # # Measurement Function
        # track.kalmanFilter.H = np.array([[1., 0, 0, 0, 0, 0],
        #                                  [0, 1, 0, 0, 0, 0]])
        # # Covariance Matrix
        # track.kalmanFilter.P = np.diag([200., 200, 50, 50, 10, 10])
        # # Motion Noise
        # track.kalmanFilter.Q = np.diag([100., 100, 25, 25, 50, 50])
        # # Measurement Noise
        # track.kalmanFilter.R = 100

        tracks.append(track)

        next_id += 1

    return next_id


def display_tracking_results(frame, masked, tracks, counter, out_original, out_masked):
    min_track_age = 1.0 * FPS    # seconds * FPS to give number of frames in seconds
    min_visible_count = 0.7 * FPS

    good_tracks = []

    masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)

    if len(tracks) != 0:
        for track in tracks:
            if track.age > min_track_age and track.totalVisibleCount > min_visible_count:
                centroid = track.kalmanFilter.x[:2]
                size = track.size

                good_tracks.append([track.id, track.age, size, (centroid[0], centroid[1])])

                rect_top_left = (int(centroid[0] - size/2), int(centroid[1] - size/2))
                rect_bottom_right = (int(centroid[0] + size/2), int(centroid[1] + size/2))
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

    out_original.write(frame)
    out_masked.write(masked)

    # cv2.imwrite('output/frame/' + str(counter) + '.png', frame)
    # cv2.imwrite('output/masked/' + str(counter) + '.png', masked)

    scale = 50
    window_size = (16*scale, 9*scale)
    frame = cv2.resize(frame, window_size, interpolation=cv2.INTER_CUBIC)
    masked = cv2.resize(masked, window_size, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('frame', frame)
    cv2.imshow('masked', masked)

    return good_tracks
