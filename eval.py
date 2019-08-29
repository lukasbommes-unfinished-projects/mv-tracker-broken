import os
import glob
import csv

import numpy as np
from tqdm import tqdm

from video_cap import VideoCap
from mvt.tracker import MotionVectorTracker
from config import EvalConfig as Config

import motmetrics as mm


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


if __name__ == "__main__":

    print("Evaluating datasets: {}".format(Config.EVAL_DATASETS))
    print("Evaluating with detections: {}".format(Config.EVAL_DETECTORS))

    train_dirs = sorted(glob.glob(os.path.join(Config.DATA_DIR, "train/*")))
    test_dirs = sorted(glob.glob(os.path.join(Config.DATA_DIR, "test/*")))
    data_dirs = []
    if "test" in Config.EVAL_DATASETS:
        data_dirs += test_dirs
    if "train" in Config.EVAL_DATASETS:
        data_dirs += train_dirs

    for data_dir in data_dirs:
        video_file = os.path.join(data_dir, 'seq.mp4')

        num_frames = len(glob.glob(os.path.join(data_dir, 'img1/*.jpg')))

        detections = load_detections(os.path.join(data_dir, 'det/det.txt'), num_frames)
        sequence_name = data_dir.split('/')[-1]

        detector_name = sequence_name.split('-')[-1]
        if detector_name not in Config.EVAL_DETECTORS:
            continue

        print("Computing MOT metrics for sequence {}".format(sequence_name))

        tracker = MotionVectorTracker(iou_threshold=Config.TRACKER_IOU_THRES)
        cap = VideoCap()

        ret = cap.open(video_file)
        if not ret:
            raise RuntimeError("Could not open the video file")

        frame_idx = 0

        with open(os.path.join('output', '{}.txt'.format(sequence_name)), mode="w") as csvfile:

            csv_writer = csv.writer(csvfile, delimiter=',')

            pbar = tqdm(total=len(detections))
            while True:
                ret, frame, motion_vectors, frame_type, _ = cap.read()
                if not ret:
                    break

                # update with detections
                if frame_idx % Config.DETECTOR_INTERVAL == 0:
                    tracker.update(motion_vectors, frame_type, detections[frame_idx])

                # prediction by tracker
                else:
                    tracker.predict(motion_vectors, frame_type)

                track_boxes = tracker.get_boxes()
                track_ids = tracker.get_box_ids()

                for track_box, track_id in zip(track_boxes, track_ids):
                    csv_writer.writerow([frame_idx+1, track_id, track_box[0], track_box[1],
                        track_box[2], track_box[3], -1, -1, -1, -1])

                frame_idx += 1
                pbar.update(1)

        cap.release()
        pbar.close()
