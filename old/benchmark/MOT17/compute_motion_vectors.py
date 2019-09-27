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


def get_vectors_by_source(motion_vectors, source):
    """Returns subset of motion vectors with a specified source frame.

    The source parameter of a motion vector specifies the temporal position of
    the reference (source) frame relative to the current frame. Each vector
    starts at the point (src_x, sry_y) in the source frame and points to the
    point (dst_x, dst_y) in the current frame. If the source value is for
    example -1, then the reference frame is the previous frame.

    For B frames there are motion vectors which refer macroblocks both to past
    frames and future frames. By setting the source parameter to "past" this
    method filters out motion vectors referring to future frames and returns the
    set of motion vectors which refer to past frames (e.g. the equivalent to the
    motion vectors in P frames). Similarly, by setting the value to "future"
    only vectors referring to future frames are returned.

    Args:
        motion_vectors (`numpy.ndarray`): Array of shape (N, 11) containing all
            N motion vectors inside a frame. N = 0 is allowed meaning no vectors
            are present in the frame.

        source (`ìnt` or `string`): Motion vectors with this value for their
            source parameter (the location of the reference frame) are selected.
            If "future", all motion vectors with a positive source value are
            returned (only for B-frames). If "past" all motion vectors with
            a negative source value are returned.

    Returns:
        motion_vectors (`numpy.ndarray`): Array of shape (M, 11) containing all
        M motion vectors with the specified source value. If N = 0 => M = 0
        that is an empty numpy array of shape (0, 11) is returned.
    """
    if np.shape(motion_vectors)[0] == 0:
        return motion_vectors
    else:
        if source == "past":
            idx = np.where(motion_vectors[:, 0] < 0)[0]
        elif source == "future":
            idx = np.where(motion_vectors[:, 0] > 0)[0]
        else:
            idx = np.where(motion_vectors[:, 0] == source)[0]
        return motion_vectors[idx, :]


def get_nonzero_vectors(motion_vectors):
    """Returns subset of motion vectors which have non-zero magnitude.
    """
    if np.shape(motion_vectors)[0] == 0:
        return motion_vectors
    else:
        idx = np.where(np.logical_or(motion_vectors[:, 7] != 0, motion_vectors[:, 8] != 0))[0]
        return motion_vectors[idx, :]


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


def motion_vectors_to_image(motion_vectors, frame_shape=(1920, 1080)):
    """Converts a set of motion vectors into a BGR image.

    Args:
        motion_vectors (`numpy.ndarray`): Motion vector array with shape [N, 10]
            as returned by VideoCap. The motion vector array should only contain P-vectors
            which can be filtered out by using get_vectors_by_source(motion_vectors, "past").
            Also, the reference frame should be normalized by using normalize_vectors.

        frame_shape (`tuple` of `int`): Desired (width, height) in pixels of the returned image.
            Should correspond to the size of the source footage of which the motion vectors
            where extracted.

    Returns:
        `numpy.ndarray` The motion vectors encoded as image. Image shape is (height, widht, 3)
        and channel order is BGR. The red channel contains the scaled x motion components of
        the motion vectors and the green channel the scaled y motion components. Scaled means
        the motion components are normalized to range [0, 1].
    """
    # compute necessary frame shape
    frame_shape = (1920, 1080)
    need_width = math.ceil(frame_shape[0] / 16) * 16
    need_height = math.ceil(frame_shape[1] / 16) * 16

    image = np.zeros((need_height, need_width, 3), dtype=np.float32)

    if np.shape(motion_vectors)[0] != 0:

        # get minimum and maximum values
        mvs_dst_x = motion_vectors[:, 5]
        mvs_dst_y = motion_vectors[:, 6]
        mb_w = motion_vectors[:, 1]
        mb_h = motion_vectors[:, 2]
        mvs_tl_x = (mvs_dst_x - 0.5 * mb_w).astype(np.int64)
        mvs_tl_y = (mvs_dst_y - 0.5 * mb_h).astype(np.int64)

        # compute value
        mvs_motion_x = (motion_vectors[:, 7] / motion_vectors[:, 9]).reshape(-1, 1)
        mvs_motion_y = (motion_vectors[:, 8] / motion_vectors[:, 9]).reshape(-1, 1)

        mvs_min_x = np.min(mvs_motion_x)
        mvs_max_x = np.max(mvs_motion_x)
        mvs_min_y = np.min(mvs_motion_y)
        mvs_max_y = np.max(mvs_motion_y)

        mvs_motion_x = (mvs_motion_x - mvs_min_x) / (mvs_max_x - mvs_min_x)
        mvs_motion_y = (mvs_motion_y - mvs_min_y) / (mvs_max_y - mvs_min_y)

        for i, motion_vector in enumerate(motion_vectors):
            # repeat value
            mvs_motion_x_repeated = np.repeat(np.repeat(mvs_motion_x[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)
            mvs_motion_y_repeated = np.repeat(np.repeat(mvs_motion_y[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)

            # insert repeated block into image
            image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 2] = mvs_motion_x_repeated
            image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 1] = mvs_motion_y_repeated

    # crop the image back to frame_shape
    image = image[0:frame_shape[1], 0:frame_shape[0], :]

    return image


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
    motion_vectors = motion_vectors.astype(np.float32)
    motion_vectors_grid = np.zeros((2, math.ceil(frame_shape[1]/16), math.ceil(frame_shape[0]/16)), dtype=np.float32)
    mvs_x = motion_vectors[:, 5].astype(np.int64)
    mvs_y = motion_vectors[:, 6].astype(np.int64)
    x = ((mvs_x - 8) // 16).astype(np.int64)
    y = ((mvs_y - 8) // 16).astype(np.int64)
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

    codec = "h264"  # whether to use h264 or mpeg4 video sequences

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
                    motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
                    motion_vectors = normalize_vectors(motion_vectors)
                    #motion_vectors_non_zero = get_nonzero_vectors(motion_vectors)
                    #motion_vectors_image = motion_vectors_to_image(motion_vectors_non_zero, frame_shape)
                    if codec == "h264":
                        mvs_x_interp, mvs_y_interp = interp_motion_vectors(motion_vectors, frame_shape)
                        mvs = torch.from_numpy(np.dstack((mvs_x_interp, mvs_y_interp)))
                        mvs = mvs.permute(2, 0, 1).float()  # store as C, H, W
                    elif codec == "mpeg4":
                        mvs = motion_vectors_to_grid(motion_vectors)
                        mvs = torch.from_numpy(mvs).float()
                    data_item["motion_vectors"] = mvs
                    #data_item["motion_vectors_image"] = torch.from_numpy(motion_vectors_image).float()
                else:
                    data_item["motion_vectors"] = torch.zeros([2, 68, 120], dtype=torch.float)
                    #data_item["motion_vectors_image"] = torch.zeros([frame_shape[1], frame_shape[0], 3], dtype=torch.float)

                data.append(data_item)

            cap.release()
            pbar.close()

        pickle.dump(data, open(os.path.join("preprocessed", codec, mode, "data.pkl"), 'wb'))
        pickle.dump(lengths[mode], open(os.path.join("preprocessed", codec, mode, "lengths.pkl"), 'wb'))
