import os
import time

import numpy as np
import cv2
import pickle

from video_cap import VideoCap
from mvt.utils import draw_motion_vectors, draw_boxes

from mvt.tracker import MotionVectorTracker
from mvt.detector import DetectorTF
from config import Config


if __name__ == "__main__":

    video_file = "video.avi"

    detector = DetectorTF(path=Config.DETECTOR_PATH,
                        box_size_threshold=Config.DETECTOR_BOX_SIZE_THRES,
                        scaling_factor=Config.SCALING_FACTOR,
                        gpu=0)
    tracker = MotionVectorTracker(iou_threshold=Config.TRACKER_IOU_THRES)
    cap = VideoCap()

    ret = cap.open(video_file)
    if not ret:
        raise RuntimeError("Could not open the video file")

    frame_idx = 0
    step_wise = False

    # box colors
    color_detection = (0, 0, 150)
    color_tracker = (0, 0, 255)
    color_previous = (150, 255, 0)

    while True:
        ret, frame, motion_vectors, frame_type, _ = cap.read()
        if not ret:
            print("Could not read the next frame")
            break

        print("-------------------")
        print("Frame Index: ", frame_idx)
        print("Frame type: ", frame_type)

        # draw entire field of motion vectors
        frame = draw_motion_vectors(frame, motion_vectors, color=(0, 0, 255))

        # draw info
        frame = cv2.putText(frame, "Frame Type: {}".format(frame_type), (1000, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # draw color legend
        frame = cv2.putText(frame, "Detection", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_detection, 2, cv2.LINE_AA)
        #frame = cv2.putText(frame, "Previous Prediction", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_previous, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Tracker Prediction", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tracker, 2, cv2.LINE_AA)
        #frame = cv2.putText(frame, "Desired Prediction", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_desired, 2, cv2.LINE_AA)

        # update with detections
        if frame_idx % Config.DETECTOR_INTERVAL == 0:
            detections = detector.detect(frame)
            det_boxes = detections['detection_boxes']
            tracker.update(motion_vectors, frame_type, det_boxes)

        # prediction by tracker
        else:
            tracker.predict(motion_vectors, frame_type)
            track_boxes = tracker.get_boxes()
            frame = draw_boxes(frame, track_boxes, color=color_tracker)

        frame = draw_boxes(frame, det_boxes, color=color_detection)

        frame_idx += 1
        cv2.imshow("Frame", frame)

        # handle key presses
        # 'q' - Quit the running program
        # 's' - enter stepwise mode
        # 'a' - exit stepwise mode
        key = cv2.waitKey(1)
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

    cap.release()
    cv2.destroyAllWindows()
