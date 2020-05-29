import cv2
import numpy
import time
from collections import deque
import math
import statistics

# Turns out the frame skipping was pointless because the way the frames are read already acounts for this
# Probably some lower level thing
# def real_time_processing():
#     cap = cv2.VideoCapture(0)
#
#     target_fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     print(f"Video Resolution: {frame_width} by {frame_height}")
#     print(f"Camera FPS: {target_fps}")
#
#     frame_time_history = deque(maxlen=60)
#
#     frames_to_skip = 0
#     skip_count = 0
#
#     while cap.isOpened():
#         start_time = time.time()
#
#         if skip_count >= frames_to_skip:
#             skip_count = 0
#
#             ret, frame = cap.read()
#
#             if ret:
#                 cv2.imshow('frame', frame)
#
#                 time.sleep(0.2)
#
#                 end_time = time.time()
#                 dt = end_time - start_time
#                 # frames_to_skip = dt / (1 / target_fps)
#                 print(f'skip count: {skip_count}')
#
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#
#             else:
#                 cap.release()
#                 cv2.destroyAllWindows()
#
#         else:
#             skip_count += 1
#
#         end_time = time.time()
#         dt = end_time - start_time
#
#         frame_time_history.append(dt)
#
#         print(f"fps = {1/statistics.mean(frame_time_history)}")
#
#         print(f"frames to skip: {frames_to_skip}")
#
#     # cap.release()
#     # cv2.destroyAllWindows()


if __name__ == '__main__':
    # real_time_processing()
    pass