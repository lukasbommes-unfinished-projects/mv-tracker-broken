import os
import glob
import time

import numpy as np
import cv2
import pickle
from tqdm import tqdm

import motmetrics as mm

from mvt.utils import draw_motion_vectors, draw_boxes, draw_box_ids, draw_shifts

from mvt.tracker import MotionVectorTracker
from config import Config

output_graphics = True


if __name__ == "__main__":

    acc = mm.MOTAccumulator(auto_id=True)

    tracker = MotionVectorTracker(iou_threshold=Config.TRACKER_IOU_THRES)

    step_wise = False

    # box colors
    color_detection = (0, 0, 150)
    color_tracker = (0, 0, 255)
    color_gt = (0, 255, 0)
    #color_previous = (150, 255, 0)

    if output_graphics:
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame", 640, 360)

    dev_outputs = sorted(glob.glob("dev_output/*.pkl"))
    with tqdm(total=len(dev_outputs)) as pbar:
        for frame_idx, dev_output in enumerate(dev_outputs):
            data = pickle.load(open(dev_output, "rb"))

            frame = data["frame"]
            frame_type = data["frame_type"]
            det_boxes = data["det_boxes"]
            gt_boxes = data["gt_boxes"]
            gt_ids = data["gt_ids"]
            motion_vectors = data["motion_vectors"]

            if output_graphics:
                # draw entire field of motion vectors
                frame = draw_motion_vectors(frame, motion_vectors, color=(0, 0, 255))
                # draw info
                frame = cv2.putText(frame, "Frame Type: {}".format(frame_type), (1000, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                frame = cv2.putText(frame, "Step: {}".format(frame_idx), (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                # draw color legend
                frame = cv2.putText(frame, "Last Detection", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_detection, 2, cv2.LINE_AA)
                frame = cv2.putText(frame, "Tracker Prediction", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tracker, 2, cv2.LINE_AA)
                frame = cv2.putText(frame, "Groundtruth", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_gt, 2, cv2.LINE_AA)

            # update with detections
            if frame_idx % Config.DETECTOR_INTERVAL == 0:
                tracker.update(motion_vectors, frame_type, det_boxes)

            # prediction by tracker
            else:
                tracker.predict(motion_vectors, frame_type)

            track_boxes = tracker.get_boxes()
            track_ids = tracker.get_box_ids()
            shifts = tracker.get_shifts()

            dist = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
            acc.update(gt_ids, track_ids, dist)

            frame_idx += 1

            if output_graphics:
                frame = draw_boxes(frame, det_boxes, color=color_detection)
                frame = draw_boxes(frame, track_boxes, color=color_tracker)
                frame = draw_boxes(frame, gt_boxes, color=color_gt)
                frame = draw_shifts(frame, shifts, track_boxes)
                frame = draw_box_ids(frame, track_boxes, track_ids)
                frame = draw_box_ids(frame, det_boxes, range(len(det_boxes)))

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

            pbar.update()

    if output_graphics:
        cv2.destroyAllWindows()

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
