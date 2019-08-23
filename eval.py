import os
import time
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from video_cap import VideoCap
from mvt.utils import draw_motion_vectors, draw_boxes, draw_box_ids

from mvt.tracker import MotionVectorTracker
from config import EvalConfig as Config

import motmetrics as mm


def load_detections(det_file):
    detections = []
    raw_data = np.genfromtxt(det_file, delimiter=',')
    for frame_idx in set(raw_data[:, 0]):
        idx = np.where(raw_data[:, 0] == frame_idx)
        detections.append(np.stack(raw_data[idx], axis=0)[:, 2:6])
    return detections

def load_groundtruth(gt_file):
    gt_boxes = []
    gt_ids = []
    raw_data = np.genfromtxt(gt_file, delimiter=',')
    for frame_idx in set(raw_data[:, 0]):
        idx = np.where(raw_data[:, 0] == frame_idx)
        gt_boxes.append(np.stack(raw_data[idx], axis=0)[:, 2:6])
        gt_ids.append(np.stack(raw_data[idx], axis=0)[:, 1])
    return gt_ids, gt_boxes


if __name__ == "__main__":

    accs = []
    for seq_idx in ["02", "04", "05", "09", "10", "11", "13"]:
        video_file = os.path.join(Config.DATA_DIR, 'train', 'MOT17-{}-{}/seq.avi'.format(seq_idx, Config.DET_TYPE))
        detections = load_detections(os.path.join(Config.DATA_DIR, 'train', 'MOT17-{}-{}/det/det.txt'.format(seq_idx, Config.DET_TYPE)))
        gt_ids, gt_boxes = load_groundtruth(os.path.join(Config.DATA_DIR, 'train', 'MOT17-{}-{}/gt/gt.txt'.format(seq_idx, Config.DET_TYPE)))

        print("Computing MOT metrics for sequence MOT17-{}-{}".format(seq_idx, Config.DET_TYPE))

        acc = mm.MOTAccumulator(auto_id=True)

        tracker = MotionVectorTracker(iou_threshold=Config.TRACKER_IOU_THRES)
        cap = VideoCap()

        ret = cap.open(video_file)
        if not ret:
            raise RuntimeError("Could not open the video file")

        frame_idx = 0
        step_wise = False

        pbar = tqdm(total=len(detections))
        while True:
            ret, frame, motion_vectors, frame_type, _ = cap.read()
            if not ret:
                break

            # draw entire field of motion vectors
            frame = draw_motion_vectors(frame, motion_vectors, color=(0, 0, 255))

            # draw info
            frame = cv2.putText(frame, "Frame Type: {}".format(frame_type), (1000, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # update with detections
            if frame_idx % Config.DETECTOR_INTERVAL == 0:
                tracker.update(motion_vectors, frame_type, detections[frame_idx])

            # prediction by tracker
            else:
                tracker.predict(motion_vectors, frame_type)

            track_boxes = tracker.get_boxes()
            track_ids = tracker.get_box_ids()
            frame = draw_boxes(frame, track_boxes, color=(0, 0, 255))
            frame = draw_box_ids(frame, track_boxes, track_ids)

            # compute distance between ground truth and tracker prediction
            dist = mm.distances.iou_matrix(track_boxes, gt_boxes[frame_idx], max_iou=0.5)
            acc.update(gt_ids[frame_idx], track_ids, dist)

            frame_idx += 1
            pbar.update(1)
            # cv2.imshow("Frame", frame)
            #
            # # handle key presses
            # # 'q' - Quit the running program
            # # 's' - enter stepwise mode
            # # 'a' - exit stepwise mode
            # key = cv2.waitKey(1)
            # if not step_wise and key == ord('s'):
            #     step_wise = True
            # if key == ord('q'):
            #     break
            # if step_wise:
            #     while True:
            #         key = cv2.waitKey(1)
            #         if key == ord('s'):
            #             break
            #         elif key == ord('a'):
            #             step_wise = False
            #             break

        cap.release()
        pbar.close()

        # compute metrics for this sequence
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='result')
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)

        accs.append(acc)


        cv2.destroyAllWindows()

    # TODO: compute overall metrics table
