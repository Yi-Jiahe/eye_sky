from object_tracker import motion_based_multi_object_tracking
import math
import matplotlib.pyplot as plt


def plot_tracks(tracks):
    # max_id = tracks[-1][-1][0]
    frame_count = len(tracks)-1

    for frame in tracks[1:]:
        for track in frame:
            track_id = track[0]
            hex_code = scalar_to_hex(tracks.index(frame), frame_count)
            plt.scatter(track[3][0], -track[3][1], c=hex_code, marker='+')
    plt.xlim(0, tracks[0][0])
    plt.ylim(-tracks[0][1], 0)
    plt.show()


def scalar_to_hex(track_id, max_id):
    f = track_id/max_id
    a = (1-f)*5
    x = math.floor(a)
    y = math.floor(255*(a-x))
    if x == 0:
        return '#%02x%02x%02x' % (255, y, 0)
    elif x == 1:
        return '#%02x%02x%02x' % (255, 255, 0)
    elif x == 2:
        return '#%02x%02x%02x' % (0, 255, y)
    elif x == 3:
        return '#%02x%02x%02x' % (0, 255, 255)
    elif x == 4:
        return '#%02x%02x%02x' % (y, 0, 255)
    else: # x == 5:
        return '#%02x%02x%02x' % (255, 0, 255)


plot_tracks(motion_based_multi_object_tracking('pan_zoom_sky.mp4'))