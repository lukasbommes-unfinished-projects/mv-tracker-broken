import os
import glob
import time

import numpy as np
import cv2
import pickle

from video_cap import VideoCap
from mvt.utils import draw_motion_vectors, draw_boxes, draw_box_ids, draw_shifts

from mvt.tracker import MotionVectorTracker
from config import Config


def load_detections(det_file, num_frames):
    detections = []
    raw_data = np.genfromtxt(det_file, delimiter=',')
    for frame_idx in range(num_frames):
        idx = np.where(raw_data[:, 0] == frame_idx+1)
        if idx[0].size:
            detections.append(np.stack(raw_data[idx], axis=0)[:, 2:6])
        else:
            detections.append(np.empty(shape=(0,10)))
    return detections


def load_groundtruth(gt_file, only_eval=False):
    """
    Args:
        gt_file (string): Full path of a MOT groundtruth txt file.

        only_eval (bool): If False load all groundtruth entries, otherwise
            load only entries in which column 7 is 1 indicating an entry that
            is to be considered during evaluation.
    """
    gt_boxes = []
    gt_ids = []
    gt_classes = []
    raw_data = np.genfromtxt(gt_file, delimiter=',')
    for frame_idx in sorted(set(raw_data[:, 0])):
        idx = np.where(raw_data[:, 0] == frame_idx)
        gt_box = np.stack(raw_data[idx], axis=0)[:, 2:6]
        gt_id = np.stack(raw_data[idx], axis=0)[:, 1]
        gt_class = np.stack(raw_data[idx], axis=0)[:, 7]
        consider_in_eval = np.stack(raw_data[idx], axis=0)[:, 6]
        consider_in_eval = consider_in_eval.astype(np.bool)
        if only_eval:
            gt_box = gt_box[consider_in_eval]
            gt_id = gt_id[consider_in_eval]
        gt_boxes.append(gt_box)
        gt_ids.append(gt_id)
        gt_classes.append(gt_class)
    return gt_ids, gt_boxes, gt_classes


if __name__ == "__main__":

    root_dir = "benchmark/MOT17/train"
    sequence = "MOT17-09"
    detector = "FRCNN"
    codec = "h264"

    data_dir = os.path.join(root_dir, "{}-{}".format(sequence, detector))

    num_frames = len(glob.glob(os.path.join(data_dir, 'img1/*.jpg')))
    detections = load_detections(os.path.join(data_dir, 'det/det.txt'), num_frames)
    gt_ids, gt_boxes, gt_classes = load_groundtruth(os.path.join(data_dir, 'gt/gt.txt'), only_eval=True)

    tracker = MotionVectorTracker(iou_threshold=Config.TRACKER_IOU_THRES)
    cap = VideoCap()

    if codec == "h264":
        ret = cap.open(os.path.join(root_dir, "..", "sequences", "h264", "{}.mp4".format(sequence)))
    elif codec == "mpeg4":
        ret = cap.open(os.path.join(root_dir, "..", "sequences", "mpeg4", "{}.avi".format(sequence)))
    if not ret:
        raise RuntimeError("Could not open the video file")

    frame_idx = 0
    step_wise = False

    # box colors
    color_detection = (0, 0, 150)
    color_tracker = (0, 0, 255)
    color_gt = (0, 255, 0)
    #color_previous = (150, 255, 0)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 640, 360)

    while True:
        ret, frame, motion_vectors, frame_type, _ = cap.read()
        if not ret:
            print("Could not read the next frame")
            break

        # draw entire field of motion vectors
        frame = draw_motion_vectors(frame, motion_vectors)

        # draw info
        frame = cv2.putText(frame, "Frame Type: {}".format(frame_type), (1000, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Step: {}".format(frame_idx), (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # draw color legend
        frame = cv2.putText(frame, "Last Detection", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_detection, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Tracker Prediction", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tracker, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Groundtruth", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_gt, 2, cv2.LINE_AA)
        #frame = cv2.putText(frame, "Previous Prediction", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_previous, 2, cv2.LINE_AA)

        # update with detections
        if frame_idx % Config.DETECTOR_INTERVAL == 0:
            det_boxes = detections[frame_idx]
            tracker.update(motion_vectors, frame_type, det_boxes)

        # prediction by tracker
        else:
            tracker.predict(motion_vectors, frame_type)

        track_boxes = tracker.get_boxes()
        track_ids = tracker.get_box_ids()
        tracker_debug_data = tracker.get_debug_data()

        frame = draw_boxes(frame, det_boxes, color=color_detection)
        frame = draw_boxes(frame, track_boxes, color=color_tracker)
        frame = draw_boxes(frame, gt_boxes[frame_idx], color=color_gt)
        frame = draw_box_ids(frame, track_boxes, track_ids)
        frame = draw_box_ids(frame, det_boxes, range(len(detections)))

        if tracker_debug_data["type"] == "predict":
            frame = draw_shifts(frame, tracker_debug_data["shifts"], track_boxes)

        # write data to file
        # dev_output = {
        #     "frame": frame,
        #     "det_boxes": det_boxes,
        #     "gt_ids": gt_ids[frame_idx],
        #     "gt_boxes": gt_boxes[frame_idx],
        #     "motion_vectors": motion_vectors,
        #     "tracker_debug_data": tracker_debug_data,
        #     "frame_type": frame_type
        # }
        # pickle.dump(dev_output, open("dev_output/{:08d}.pkl".format(frame_idx), "wb"))

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
