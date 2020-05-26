import cv2
import math
import numpy as np
import time


def imshow_resized(window_name, img):
    window_size = (int(848), int(480))
    img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, img)


def stabilize_frame(img1_in, img2_in):
    img1_gray = cv2.cvtColor(img1_in, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_in, cv2.COLOR_BGR2GRAY)

    image_points1 = cv2.goodFeaturesToTrack(img1_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    image_points2, status, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, image_points1, None)

    # Filtering bad points
    idx = np.where(status == 1)[0]
    image_points1 = image_points1[idx]
    image_points2 = image_points2[idx]

    # Estimate affine transformation
    # Produces a transformation matrix :
    # [[cos(theta).s, -sin(theta).s, tx],
    # [sin(theta).s, cos(theta).s, ty]]
    # where theta is rotation, s is scaling and tx,ty are translation
    m = cv2.estimateAffinePartial2D(image_points1, image_points2)[0]

    # im = cv2.invertAffineTransform(m)

    # Extract translation
    dx = m[0, 2]
    dy = m[1, 2]
    # print(f"dx={dx}, dy={dy}")
    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])

    rows, columns = img2_gray.shape
    frame_stabilized = cv2.warpAffine(img2_in, m, (columns, rows))

    return frame_stabilized, dx, dy


def stabilize_frame_standalone(filename):
    cap = cv2.VideoCapture(filename)

    global FRAME_WIDTH, FRAME_HEIGHT
    FRAME_WIDTH = int(cap.get(3))
    FRAME_HEIGHT = int(cap.get(4))
    print(f"Video Resolution: {FRAME_WIDTH} by {FRAME_HEIGHT}")

    global_height, global_width = FRAME_HEIGHT, FRAME_WIDTH
    global_frame = np.zeros((global_height, global_width, 3), dtype=np.uint8)
    top_left, bottom_right = [0, 0], [FRAME_WIDTH, FRAME_HEIGHT]

    frame_count = 0
    scene_transition = False

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            imshow_resized('original', frame)

            if frame_count == 0:
                frame_before = frame
            elif frame_count >= 1:
                image1 = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
                image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                image_points1 = cv2.goodFeaturesToTrack(image1, maxCorners=100, qualityLevel=0.01, minDistance=10)

                image_points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, image_points1, None)

                idx = np.where(status == 1)[0]
                image_points1 = image_points1[idx]
                image_points2 = image_points2[idx]

                # Estimate affine transformation
                # Produces a transformation matrix :
                # [[cos(theta).s, -sin(theta).s, tx],
                # [sin(theta).s, cos(theta).s, ty]]
                # where theta is rotation, s is scaling and tx,ty are translation
                m = cv2.estimateAffinePartial2D(image_points1, image_points2)[0]

                # im = cv2.invertAffineTransform(m)

                # Extract translation
                dx = m[0, 2]
                dy = m[1, 2]
                # print(f"dx={dx}, dy={dy}")
                # Extract rotation angle
                da = np.arctan2(m[1, 0], m[0, 0])

                movement_threshold = 2
                if scene_transition == False:
                    if math.fabs(dx) > movement_threshold or math.fabs(dy) > movement_threshold:
                        scene_transition = True
                        print('Moving')
                else:   # scene_transition is True
                    if math.fabs(dx) <= movement_threshold and math.fabs(dy) <= movement_threshold:
                        scene_transition = False
                        print('Stopped')

                top_left[0] -= int(dx)
                bottom_right[0] -= int(dx)
                top_left[1] -= int(dy)
                bottom_right[1] -= int(dy)

                if top_left[0] < 0:
                    pad_columns = np.zeros((global_height, int(math.fabs(dx)), 3), dtype=np.uint8)
                    global_frame = np.hstack((pad_columns, global_frame))
                    global_width += int(math.fabs(dx))
                    top_left[0] = 0
                    bottom_right[0] += int(math.fabs(dx))
                elif bottom_right[0] > global_width:
                    pad_columns = np.zeros((global_height, int(math.fabs(dx)), 3), dtype=np.uint8)
                    global_frame = np.hstack((global_frame, pad_columns))
                    global_width += int(math.fabs(dx))
                else:
                    pass
                if top_left[1] < 0:
                    pad_rows = np.zeros((int(math.fabs(dy)), global_width, 3), dtype=np.uint8)
                    global_frame = np.vstack((pad_rows, global_frame))
                    global_height += int(math.fabs(dy))
                    top_left[1] = 0
                    bottom_right[1] += int(math.fabs(dy))
                elif bottom_right[1] > global_height:
                    pad_rows = np.zeros((int(math.fabs(dy)), global_width, 3), dtype=np.uint8)
                    global_frame = np.vstack((global_frame, pad_rows))
                    global_height += int(math.fabs(dy))
                else:
                    pass

                rows, columns = image1.shape
                frame_stabilized = cv2.warpAffine(frame, m, (columns, rows))

                frame_before = frame
                frame = frame_stabilized

            global_frame[top_left[1]:top_left[1]+FRAME_HEIGHT, top_left[0]:top_left[0]+FRAME_WIDTH, :] = frame
            print(f"Top left: {top_left}, Bottom Right: {bottom_right}")
            
            display_frame = global_frame.copy()
            cv2.rectangle(display_frame, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]),
                          (0, 255, 255), 3)

            imshow_resized('stabilized', frame)
            imshow_resized('big picture', display_frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    stabilize_frame_standalone('panning_video.mp4')