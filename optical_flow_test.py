import cv2
import numpy as np
from camera_stabilizer import Camera
from object_tracker import imshow_resized


def imshow_resized(window_name, img):
    window_size = (int(1366), int(768))
    img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, img)


def show_optical_flow(filename):
    cap = cv2.VideoCapture(filename)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    global_height, global_width = frame_height, frame_width

    camera = Camera([frame_width, frame_height])

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame, mask = camera.undistort(frame)

            if frame_count == 0:
                frame_before = frame
            elif frame_count >= 1:
                image1 = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
                image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                image_points1 = cv2.goodFeaturesToTrack(image1, maxCorners=300, qualityLevel=0.01, minDistance=10,
                                                        mask=mask)

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

                rows, columns = image1.shape
                frame_stabilized = cv2.warpAffine(frame, m, (columns, rows))

                frame_before = frame
                frame = frame_stabilized

                for point in image_points1:
                    x, y = point.ravel()
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                for point in image_points2:
                    x, y = point.ravel()
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                imshow_resized('optical_flow', frame)

            frame_count += 1

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


show_optical_flow('tello_360_pan.mp4')