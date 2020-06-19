import cv2
import numpy as np

# cap = cv2.VideoCapture('drone_on_background/DJI_0022_1.MOV')
cap = cv2.VideoCapture('drone_on_background/GoPro_GH010763.mp4')
# cap = cv2.VideoCapture('drone_on_background/Samsung_v2.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
# fgbg = cv2.createBackgroundSubtractorKNN()

consecutive_dropped_frames = 0
consecutive_dropped_frames_tolerance = 4
while cap.isOpened():
    ret, frame = cap.read()

    print(ret, frame)

    if ret:
        consecutive_dropped_frames = 0

        # frame = cv2.resize(frame, (848, 480))

        fgmask = fgbg.apply(frame)

        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)

        fgmask = cv2.dilate(fgmask, kernel, iterations=3)

        cv2.imshow('fgmask', fgmask)
        cv2.imshow('original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print('No frame!')
        if consecutive_dropped_frames >= consecutive_dropped_frames_tolerance:
            print('Terminating read')
            break
        consecutive_dropped_frames += 1
        continue

cap.release()
cv2.destroyAllWindows()