import cv2
import numpy as np
import json
from object_tracker import imshow_resized


class CameraTest:
    def __init__(self):
        # Generic matrices, perform calibration if possible
        self.cameraMatrix = np.zeros((3, 3))
        self.distortionCoefficients = np.zeros((5))

    def calibrate_camera(self, calibration_images):
        # Termination criteria for cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points
        # This creates a 2-D set of regularly spaced points positioned in 3-D such that z=0
        # i.e. it assumes that the calibration pattern is regular and flat
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points of object in real world space (Its going to all be the same since we assume the same
                        # object is used for the calibration)
        imgpoints = []  # 2d points in image plane

        for image in calibration_images:
            frame = cv2.imread(image)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find chessboard
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            if ret:
                print('corners found')

                objpoints.append(objp)

                corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

                imgpoints.append(corners_refined)

                print('done')

                img = cv2.drawChessboardCorners(frame, (7, 6), corners_refined, ret)

                imshow_resized(image, img)
                cv2.waitKey(500)

            else:
                print('failed')

        ret, self.cameraMatrix, self.distortionCoefficients, _, _ = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


        print('finished')

        cv2.destroyAllWindows()

        self.export_calibration_results()

    def import_calibration_results(self, filename):
        with open(filename) as file:
            data = json.loads(file.read())
            self.cameraMatrix = np.array(data['camera_matrix'])
            self.distortionCoefficients = np.array(data['distortion_coefficients'])
        print(f"Camera matrix: \n {np.array(data['camera_matrix'])}\n"
              f"Distortion Coefficients: \n {np.array(data['distortion_coefficients'])}")

    def export_calibration_results(self):
        with open('camera_parameters.txt', 'w+') as file:
            file.write(json.dumps({"camera_matrix":  self.cameraMatrix.tolist(),
                                   "distortion_coefficients": self.distortionCoefficients.tolist()}))

    def undistort(self, frame):
        # getOptimalNewCameraMatrix()
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distortionCoefficients,
                                                          (w, h), 1, (w, h))

        frame_undistorted = cv2.undistort(frame, self.cameraMatrix, self.distortionCoefficients, None, newcameramtx)

        mask = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(mask, contours, -1, 255, -1)

        # crop the image
        x, y, w, h = roi
        dst = frame_undistorted[y:y + h, x:x + w]

        imshow_resized("undistorted", dst)


images = ['DJI_0022.JPG',
          'DJI_0023.JPG',
          'DJI_0024.JPG',
          'DJI_0025.JPG',
          'DJI_0027.JPG',
          'DJI_0028.JPG',
          'DJI_0029.JPG',
          'DJI_0030.JPG',
          'DJI_0031.JPG']

