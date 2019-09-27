import os
import glob
import math
import pickle

import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata

from video_cap import VideoCap


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


def normalize_vectors(motion_vectors):
    """Normalizes motion vectors to the past frame as reference frame.

    The source value in the first column is set to -1 for all frames. The x and
    y motion values are scaled accordingly. Vector source position and
    destination position are unchanged.

    Args:
        motion_vectors (`numpy.ndarray`): Array of shape (N, 11) containing all
            N motion vectors inside a frame. N = 0 is allowed meaning no vectors
            are present in the frame.

    Returns:
        motion_vectors (`numpy.ndarray`): Array of shape (M, 11) containing the
        normalized motion vectors. If N = 0 => M = 0 that is an empty numpy
        array of shape (0, 11) is returned.
    """
    if np.shape(motion_vectors)[0] == 0:
        return motion_vectors
    else:
        motion_vectors[:, 7] = motion_vectors[:, 7] / motion_vectors[:, 0]  # motion_x
        motion_vectors[:, 8] = motion_vectors[:, 8] / motion_vectors[:, 0]  # motion_y
        motion_vectors[:, 0] = -1 * np.ones_like(motion_vectors[:, 0])
        return motion_vectors


def interp_motion_vectors(motion_vectors, frame_shape=(1920, 1080)):
    mvs_x = motion_vectors[:, 5]
    mvs_y = motion_vectors[:, 6]
    mvs_x_motion = motion_vectors[:, 7] / motion_vectors[:, 9]
    mvs_y_motion = motion_vectors[:, 8] / motion_vectors[:, 9]
    # takes 5.72 ms (average of 1000 runs)
    xi = np.arange(8, frame_shape[0]+1, 16)
    yi = np.arange(8, frame_shape[1]+1, 16)
    mvs_x_motion_interp = griddata((mvs_x, mvs_y), mvs_x_motion, (xi[None, :], yi[:, None]), method='nearest')
    mvs_y_motion_interp = griddata((mvs_x, mvs_y), mvs_y_motion, (xi[None, :], yi[:, None]), method='nearest')
    return mvs_x_motion_interp, mvs_y_motion_interp


def motion_vectors_to_grid(motion_vectors, frame_shape=(1920, 1080)):
    """Converts motion vectors list into 3D matrix."""
    motion_vectors_grid = np.zeros((2, math.ceil(frame_shape[1]/16), math.ceil(frame_shape[0]/16)))
    mvs_x = motion_vectors[:, 5]
    mvs_y = motion_vectors[:, 6]
    x = (mvs_x - 8) // 16
    y = (mvs_y - 8) // 16
    motion_vectors_grid[0, y, x] = motion_vectors[:, 7] / motion_vectors[:, 9]  # x component
    motion_vectors_grid[1, y, x] = motion_vectors[:, 8] / motion_vectors[:, 9] # y component
    return motion_vectors_grid


if __name__ == "__main__":

    sequences = {
        "train": [
            "MOT17-02",
            "MOT17-04",
            #"MOT17-05",
            "MOT17-11",
            "MOT17-13",
        ],
        "val": [
            "MOT17-09",
            "MOT17-10",
        ],
        "test": [
            "MOT17-01",
            "MOT17-03",
            "MOT17-06",
            "MOT17-07",
            "MOT17-08",
            "MOT17-12",
            "MOT17-14",
        ]
    }

    frame_shapes = {
        "train": [(1920, 1080),  # MOT17-02
                  (1920, 1080),  # MOT17-04
                  #(640, 480),   # MOT17-05
                  (1920, 1080),  # MOT17-11
                  (1920, 1080)], # MOT17-13
        "val": [(1920, 1080),    # MOT17-09
                (1920, 1080)],    # MOT17-10
        "test": [(1920, 1080),   # MOT17-01
                 (1920, 1080),   # MOT17-03
                 #(640, 480),    # MOT17-06
                 (1920, 1080),   # MOT17-07
                 (1920, 1080),   # MOT17-08
                 (1920, 1080),   # MOT17-12
                 (1920, 1080)]   # MOT17-14
    }

    lengths = {
        "train": [600,  # MOT17-02
                  1050,  # MOT17-04
                  #837,   # MOT17-05
                  900,  # MOT17-11
                  750], # MOT17-13
        "val": [525,    # MOT17-09
                654],    # MOT17-10
        "test": [450,   # MOT17-01
                 1500,   # MOT17-03
                 #1194,    # MOT17-06
                 500,   # MOT17-07
                 625,   # MOT17-08
                 900,   # MOT17-12
                 750]   # MOT17-14
    }

    codec = "mpeg4"  # whether to use h264 or mpeg4 video sequences

    for mode in ["train", "val", "test"]:

        data = []

        for sequence, frame_shape in zip(sequences[mode], frame_shapes[mode]):

            if mode == "val":
                dirname = os.path.join("train", "{}-FRCNN".format(sequence))
            else:
                dirname = os.path.join(mode, "{}-FRCNN".format(sequence))

            if codec == "h264":
                video_file = os.path.join("sequences", "h264", "{}.mp4".format(sequence))
            elif codec == "mpeg4":
                video_file = os.path.join("sequences", "mpeg4", "{}.avi".format(sequence))

            num_frames = len(glob.glob(os.path.join(dirname, 'img1/*.jpg')))
            detections = load_detections(os.path.join(dirname, 'det/det.txt'), num_frames)
            if mode == "train" or mode == "val":
                gt_ids, gt_boxes, _ = load_groundtruth(os.path.join(dirname, 'gt/gt.txt'), only_eval=True)

            print("Extracting motion vectors for sequence {}".format(sequence))

            cap = VideoCap()
            ret = cap.open(video_file)
            if not ret:
                raise RuntimeError("Could not open the video file")

            pbar = tqdm(total=num_frames)
            for frame_idx in range(0, num_frames):
                ret, frame, motion_vectors, frame_type, _ = cap.read()
                if not ret:
                    break

                pbar.update()

                data_item = {
                    "frame_idx": frame_idx,
                    "sequence": sequence,
                    "frame_type": frame_type,
                    "det_boxes": detections[frame_idx]
                }

                # bounding boxes
                if mode == "train" or mode == "val":
                    data_item["gt_boxes"] = gt_boxes[frame_idx]
                    data_item["gt_ids"] = gt_ids[frame_idx]
                else:
                    data_item["gt_boxes"] = []
                    data_item["gt_ids"] = []


                # motion vectors (interpolated on regular 16x16 grid)
                if frame_type != "I":
                    motion_vectors = normalize_vectors(motion_vectors)
                    if codec == "h264":
                        mvs_x_interp, mvs_y_interp = interp_motion_vectors(motion_vectors, frame_shape)
                        mvs = torch.from_numpy(np.dstack((mvs_x_interp, mvs_y_interp)))
                        mvs = mvs.permute(2, 0, 1).float()  # store as C, H, W
                    elif codec == "mpeg4":
                        mvs = motion_vectors_to_grid(motion_vectors)
                        mvs = torch.from_numpy(mvs).float()
                    data_item["motion_vectors"] = mvs
                else:
                    data_item["motion_vectors"] = torch.zeros([2, 68, 120], dtype=torch.float)

                data.append(data_item)

            cap.release()
            pbar.close()

        pickle.dump(data, open(os.path.join("preprocessed", codec, mode, "data.pkl"), 'wb'))
        pickle.dump(lengths[mode], open(os.path.join("preprocessed", codec, mode, "lengths.pkl"), 'wb'))
