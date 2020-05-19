import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, id, size):
        self.id = id
        self.size = size
        # Constant Velocity Model
        self.kalmanFilter = KalmanFilter(dim_x=4, dim_z=2)
        # # Constant Acceleration Model
        # self.kalmanFilter = KalmanFilter(dim_x=6, dim_z=2)
        self.age = 1
        self.totalVisibleCount = 1
        self.consecutiveInvisibleCount = 0


#---------------------------------------Start Section------------------------------------------#
# Implementation of imopen from Matlab:
def imopen(im_in, kernel_size, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)/(kernel_size^2)
    im_out = cv2.morphologyEx(im_in, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return im_out


# Implementation of imclose from Matlab:
def imclose(im_in, kernel_size, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)/(kernel_size^2)
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
    _,_,im_floodfill,_ = cv2.floodFill(im_th, mask, (0, 0), 255)

    # Step 3: Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Step 4: Combine the two images to get the foreground image with holes filled in
    # Floodfilled image needs to be trimmed to perform the bitwise or operation.
    # Trimming is done from the outside. I.e. the "Border" is removed
    im_out = cv2.bitwise_or(im_th, im_floodfill_inv[1:-1,1:-1])

    return im_out
#---------------------------------------End Section------------------------------------------#


def motion_based_multi_object_tracking(filename):
    cap, fgbg, detector, out_original, out_masked = setup_system_objects(filename)

    global FPS
    FPS = cap.get(cv2.CAP_PROP_FPS)

    tracks = []

    next_id = 0
    counter = 0

    centroids_log = []
    tracks_log = []

    age_id_log = []
    centroid_point_log = []

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if counter == 0:
                frame_before = frame
            elif counter >= 1:
                image1 = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
                image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                image_points1 = cv2.goodFeaturesToTrack(image1, maxCorners=100, qualityLevel=0.01, minDistance=10)

                image_points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, image_points1, None)

                idx = np.where(status == 1)[0]
                image_points1 = image_points1[idx]
                image_points2 = image_points2[idx]

                m = cv2.estimateAffinePartial2D(image_points1, image_points2)[0]
                rows, columns = image1.shape
                frame_stabilized = cv2.warpAffine(frame, m, (columns, rows))

                frame_before = frame
                frame = frame_stabilized

            centroids, sizes, masked = detect_objects(frame, fgbg, detector)
            centroids_log.append(centroids)

            predict_new_locations_of_tracks(tracks)

            assignments, unassigned_tracks, unassigned_detections = detection_to_track_assignment(tracks, centroids)

            update_assigned_tracks(assignments, tracks, centroids, sizes)
            tracks_log.append(tracks)

            update_unassigned_tracks(unassigned_tracks, tracks)
            tracks = delete_lost_tracks(tracks)
            next_id = create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes)

            display_tracking_results(frame, masked, tracks, counter, out_original, out_masked)

            age_id_log.append([])
            centroid_point_log.append([])
            for i in range(len(tracks)):
                track = tracks[i]

                age_id_log[counter].append([track.id, track.age])
                centroid_point_log[counter].append(track.kalmanFilter.x[:2])

            counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    out_original.release()
    cv2.destroyAllWindows()

    return centroid_point_log


def setup_system_objects(filename):
    cap = cv2.VideoCapture(filename)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=32, detectShadows=False)
    fgbg.setBackgroundRatio(0.3)
    fgbg.setNMixtures(5)

    params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.minArea = 1
    # params.maxArea = 1000
    detector = cv2.SimpleBlobDetector_create(params)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_original = cv2.VideoWriter('out_original.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (frame_width, frame_height))
    out_masked = cv2.VideoWriter('out_masked.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (frame_width, frame_height))

    return cap, fgbg, detector, out_original, out_masked


def detect_objects(frame, fgbg, detector):
    # Adjust contrast and brightness of image to make foreground stand out more
    # alpha used to adjust contrast, where alpha < 1 reduces contrast and alpha > 1 increases it
    # beta used to increase brightness, scale of -255? to 255
    masked = cv2.convertScaleAbs(frame, alpha=2, beta=-50)
    # masked = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

    # Subtract Background
    # Learning rate affects how aggressively the algorithm applies the changes to background ratio and stuff
    # Or so I believe. Adjust it alongside background ratio and history to tune
    masked = fgbg.apply(masked, learningRate=0.1)

    # Invert frame such that black pixels are foreground
    masked = cv2.bitwise_not(masked)

    # Close to remove black spots
    masked = imclose(masked, 3, 2)
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
        # predictedCentroid = track.kalmanFilter.x[:2]


def detection_to_track_assignment(tracks, centroids):
    n, m = len(tracks), len(centroids)
    k, l = min(n, m), max(n, m)

    cost = np.zeros((l * 2, l * 2))

    for i in range(len(tracks)):
        track = tracks[i]
        track_location = track.kalmanFilter.x[:2]
        cost[i, :m] = np.array([distance.euclidean(track_location, centroid) for centroid in centroids])

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
    detection_padding = np.ones((l, l + extra_tracks)) * unassigned_track_cost
    cost[:l, m:] = detection_padding

    # Padding cost matrix with dummy rows to account for unassigned detections
    track_padding = np.ones((l + extra_detections, l)) * unassigned_detection_cost
    cost[n:, :l] = track_padding

    row_ind, col_ind = linear_sum_assignment(cost)
    assignments_all = np.column_stack((row_ind, col_ind))

    assignments = assignments_all[(assignments_all < [n, m]).all(axis=1)]
    unassigned_tracks = assignments_all[
        (assignments_all >= [0, m]).all(axis=1) & (assignments_all < [l, l * 2]).all(axis=1)]
    unassigned_detections = assignments_all[
        (assignments_all >= [n, 0]).all(axis=1) & (assignments_all < [l * 2, l]).all(axis=1)]

    return assignments, unassigned_tracks, unassigned_detections


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


def update_unassigned_tracks(unassigned_tracks, tracks):
    for unassignedTrack in unassigned_tracks:
        track_idx = unassignedTrack[0]

        track = tracks[track_idx]

        track.age += 1
        track.consecutiveInvisibleCount += 1


def delete_lost_tracks(tracks):
    if len(tracks) == 0:
        return tracks

    invisible_for_too_long = 3 * FPS
    age_threshold = 1 * FPS

    tracks_to_be_removed = []

    for track in tracks:
        visibility = track.totalVisibleCount/track.age
        if (track.age < age_threshold and visibility < 0.8) \
                or track.consecutiveInvisibleCount >= invisible_for_too_long:
            tracks_to_be_removed.append(track)

    tracks = [track for track in tracks if track not in tracks_to_be_removed]

    return tracks


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
    min_track_age = 0 * FPS    # seconds * FPS to give number of frames in seconds
    min_visible_count = 0 * FPS

    masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)

    if len(tracks) != 0:
        for track in tracks:
            if track.age > min_track_age and track.totalVisibleCount > min_visible_count:
                centroid = track.kalmanFilter.x[:2]
                size = track.size
                rectTopLeft =(int(centroid[0] - size/2), int(centroid[1] - size/2))
                rectBottomRight = (int(centroid[0] + size/2), int(centroid[1] + size/2))
                colour = (0, 0, 255)
                thickness = 1
                cv2.rectangle(frame, rectTopLeft, rectBottomRight, colour, thickness)
                cv2.rectangle(masked, rectTopLeft, rectBottomRight, colour, thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(frame, str(track.id), (rectBottomRight[0], rectTopLeft[1]),
                            font, font_scale, colour, thickness, cv2.LINE_AA)
                cv2.putText(masked, str(track.id), (rectBottomRight[0], rectTopLeft[1]),
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


motion_based_multi_object_tracking('Kimhoe_phone.mp4')
