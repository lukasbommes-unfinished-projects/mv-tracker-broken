import cv2
import numpy as np
from video_cap import VideoCap

def draw_motion_vectors(frame, motion_vectors):
    if np.shape(motion_vectors)[0] > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
            if mv[0, 0] < 0:
                cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.3)
            else:
                cv2.arrowedLine(frame, start_pt, end_pt, (0, 255, 0), 1, cv2.LINE_AA, 0, 0.3)
    return frame

if __name__ == "__main__":

    #video_file = "data/MOT15/train/ADL-Rundle-6/ADL-Rundle-6-h264.mp4"  # OK
    #video_file = "data/MOT15/train/ADL-Rundle-8/ADL-Rundle-8-h264.mp4"  # OK
    #video_file = "data/MOT15/train/ETH-Bahnhof/ETH-Bahnhof-h264.mp4"  # OK
    #video_file = "data/MOT15/train/ETH-Pedcross2/ETH-Pedcross2-h264.mp4"  # OK
    #video_file = "data/MOT15/train/ETH-Sunnyday/ETH-Sunnyday-h264.mp4"  # OK
    #video_file = "data/MOT15/train/KITTI-13/KITTI-13-h264.mp4"  # broken
    #video_file = "data/MOT15/train/KITTI-17/KITTI-17-h264.mp4"  # broken
    #video_file = "data/MOT15/train/PETS09-S2L1/PETS09-S2L1-h264.mp4"  # OK
    #video_file = "data/MOT15/train/TUD-Campus/TUD-Campus-h264.mp4"  # OK
    #video_file = "data/MOT15/train/TUD-Stadtmitte/TUD-Stadtmitte-h264.mp4"  # OK
    #video_file = "data/MOT15/train/Venice-2/Venice-2-h264.mp4"  # OK

    video_file = "data/MOT15/test/KITTI-19/KITTI-19-h264.mp4"  # broken


    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("frame", 640, 360)

    step_wise = False

    cap = VideoCap()
    ret = cap.open(video_file)
    if not ret:
        raise RuntimeError("Could not open the video file")

    cap_cv = cv2.VideoCapture(video_file)

    while True:

        ret, frame, motion_vectors, frame_type, _ = cap.read()
        if not ret:
            print("Could not read the next frame")
            break

        ret, frame_cv = cap_cv.read()
        if not ret:
            print("Could not read the next frame")
            break

        print(np.shape(frame))
        print(np.shape(frame_cv))


        #frame = draw_motion_vectors(frame, motion_vectors)

        cv2.imshow("frame", frame)
        cv2.imshow("frame_cv", frame_cv)
        key = cv2.waitKey(25)
        if not step_wise and key == ord('s'):
            step_wise = True
        if key == ord('q'):
            break
        if step_wise:
            while True:
                key = cv2.waitKey(1)
                if key == ord('s'):
                    break
                elif key == ord('a'):
                    step_wise = False
                    break
