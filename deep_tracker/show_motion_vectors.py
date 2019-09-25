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

    video_file = "../benchmark/MOT17/train/MOT17-10-FRCNN/MOT17-10-FRCNN-h264.mp4"

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 640, 360)

    step_wise = False

    cap = VideoCap()
    ret = cap.open(video_file)
    if not ret:
        raise RuntimeError("Could not open the video file")

    while True:

        ret, frame, motion_vectors, frame_type, _ = cap.read()
        if not ret:
            print("Could not read the next frame")
            break

        frame = draw_motion_vectors(frame, motion_vectors)

        cv2.imshow("frame", frame)
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
