import numpy as np
import math
import MotionBasedMultiObjectTrackingExample

def Calculate_multiple_tracks_paths3_Xj_with_matching(age_id_log, centroid_point_log):
    track_x = []
    track_y = []
    track_xy = []
    track_turning_angle = []

    for i in range(len(centroid_point_log)):                                    # Number of frames
        track_x.append([])
        track_y.append([])
        track_turning_angle.append([])
        if not centroid_point_log[i]:                                           # If the number of centroids is not 0
            for j in range(len(centroid_point_log[i])):                         # Each centroid in the frame
                track_x[i].append([centroid_point_log[i][j][0]])
                track_y[i].append([centroid_point_log[i][j][1]])

            if i >= 2:
                # Turning Angle
                dx21 = track_x[i-1][j] - track_x[i-2][j]
                dy21 = track_y[i-1][j] - track_y[i-2][j]
                dydx21 = dy21/dx21
                A1 = math.fabs(math.degrees(math.atan(dydx21)))

                dx32 = track_x[i][j] - track_x[i-1][j]
                dy32 = track_y[i][j] - track_y[i-1][j]
                dydx32 = dy32/dx32
                A2 = math.fabs(math.degrees(math.atan(dydx32)))

                track_turning_angle = math.fabs(A2-A1)


                # Curvature
                da = math.sqrt((track_x[i][j] - track_x[i-2][j])^2 + (track_y[i][j] - track_y[i-2][j])^2)
                db = math.sqrt((track_x[i-1][j] - track_x[i-2][j])^2 + (track_y[i-1][j] - track_y[i-2][j])^2)
                dc = math.sqrt((track_x[i][j] - track_x[i-1][j])^2 + (track_y[i][j] - track_y[i-1][j])^2)
                A3 = math.fabs(math.degrees(math.acos(
                    (da^2 - db^2 - dc^2) / (2*db*dc)
                )))



            if i >= 1:
                # Pace
                pass
