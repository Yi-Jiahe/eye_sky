import cv2
import numpy as np

from object_tracking_rt import imshow_resized

original = cv2.VideoCapture('stabilized_out_original.mp4')
masked = cv2.VideoCapture('stabilized_out_masked.mp4')

out_width = max(int(original.get(cv2.CAP_PROP_FRAME_WIDTH)), int(masked.get(cv2.CAP_PROP_FRAME_WIDTH)))
out_height = int(original.get(cv2.CAP_PROP_FRAME_HEIGHT)) + int(masked.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter('out_stabilized_combined.mp4', cv2.VideoWriter_fourcc(*'h264'), 30, (out_width, out_height))

while original.isOpened() and masked.isOpened():
    ret_or, frame_or = original.read()
    ret_m, frame_m = masked.read()

    if ret_or and ret_m:
        frame_out = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        frame_out[0:frame_or.shape[0], 0:frame_or.shape[1]] = frame_or
        frame_out[frame_or.shape[0]:frame_or.shape[0]+frame_m.shape[0],
                  0:frame_m.shape[1]] = frame_m
        output.write(frame_out)
        imshow_resized("out", frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

original.release()
masked.release()
output.release()