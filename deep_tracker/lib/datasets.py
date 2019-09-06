import os
import pickle
import torch
import math
import cv2
import numpy as np

from video_cap import VideoCap

from lib.utils import velocities_from_boxes, box_from_velocities
from lib.visu import draw_boxes, draw_motion_vectors, draw_velocities, \
    motion_vectors_to_image, draw_boxes_on_motion_vector_image


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


class MotionVectorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, pad_num_boxes=52, visu=False):

        self.sequences = {
            "train": [
                "MOT17/train/MOT17-02-FRCNN",
                "MOT17/train/MOT17-04-FRCNN",
                "MOT17/train/MOT17-05-FRCNN",
                "MOT17/train/MOT17-11-FRCNN",
                "MOT17/train/MOT17-13-FRCNN",
                "MOT15/train/ETH-Bahnhof",
                "MOT15/train/ETH-Sunnyday",
                "MOT15/train/KITTI-13",
                "MOT15/train/KITTI-17",
                "MOT15/train/PETS09-S2L1",
                "MOT15/train/TUD-Campus",
                "MOT15/train/TUD-Stadtmitte"
            ],
            "val": [
                "MOT17/train/MOT17-09-FRCNN",
                "MOT17/train/MOT17-10-FRCNN"
            ],
            "test": [
                "MOT17/test/MOT17-01-FRCNN",
                "MOT17/test/MOT17-03-FRCNN",
                "MOT17/test/MOT17-06-FRCNN",
                "MOT17/test/MOT17-07-FRCNN",
                "MOT17/test/MOT17-08-FRCNN",
                "MOT17/test/MOT17-12-FRCNN",
                "MOT17/test/MOT17-14-FRCNN"
            ]
        }

        self.lens = {
            "train": [600, 1050, 837, 900, 750, 1000,
                354, 340, 145, 795, 71, 179],
            "val": [654, 525],
            "test": [450, 1500, 1194, 500, 625, 900, 750]
        }

        self.root_dir = root_dir
        self.mode = mode
        self.pad_num_boxes = pad_num_boxes
        self.visu = visu

        self.gt_boxes_prev = None  # store for next iteration
        self.gt_ids_prev = None  # store for next iteration
        self.gt_ids_all = None
        self.gt_boxes_all = None

        self.caps = []
        self.is_open = []
        for _ in self.sequences[self.mode]:
            cap = VideoCap()
            self.caps.append(cap)
            self.is_open.append(False)

        self.current_seq_id = 0
        self.current_frame_idx = 0


    def __len__(self):
        return sum(self.lens[self.mode]) - 1 # -1 is needed because of idx+1 in __getitem__


    def __getitem__(self, idx):

        #TODO: load ground truth only for train and val data

        while True:

            # this block is executed only once when a new sequence starts
            if not self.is_open[self.current_seq_id]:

                # open detections and groundtruth files
                #detections_file = os.path.join(self.root_dir, self.sequences[self.current_seq_id], "det/det.txt")
                #detections = load_detections(detections_file, num_frames=self.lens[self.mode][self.current_seq_id])
                gt_file = os.path.join(self.root_dir, self.sequences[self.mode][self.current_seq_id], "gt/gt.txt")
                self.gt_ids_all, self.gt_boxes_all, _ = load_groundtruth(gt_file, only_eval=True)

                # read and store first ground truth (and detections)
                self.gt_boxes_prev = self.gt_boxes_all[0]
                self.gt_ids_prev = self.gt_ids_all[0]

                # open the video sequence and drop first frame
                sequence_name = str.split(self.sequences[self.mode][self.current_seq_id], "/")[-1]
                video_file = os.path.join(self.root_dir, self.sequences[self.mode][self.current_seq_id], "{}-h264.mp4".format(sequence_name))
                ret = self.caps[self.current_seq_id].open(video_file)
                if not ret:
                    raise RuntimeError("Could not open the video file")
                _ = self.caps[self.current_seq_id].read()

                self.is_open[self.current_seq_id] = True
                self.current_frame_idx += 1
                continue

            # when the end of the sequence is reached switch to the next one
            if self.current_frame_idx == self.lens[self.mode][self.current_seq_id]:
                self.caps[self.current_seq_id].release()
                self.is_open[self.current_seq_id] = False
                self.current_frame_idx = 0
                self.current_seq_id =+ 1
                # make sure the sequence index wraps around at the number of sequences
                if self.current_seq_id == len(self.sequences[self.mode]):
                    self.current_seq_id = 0
                continue

            # otherwise just load, process and return the next sample
            ret, frame, motion_vectors, frame_type, _ = self.caps[self.current_seq_id].read()
            if not ret:  # should never happen
                raise RuntimeError("Could not read next frame from video")

            # convert motion vectors to image (for I frame black image is returned)
            motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
            motion_vectors = normalize_vectors(motion_vectors)
            motion_vectors = get_nonzero_vectors(motion_vectors)
            motion_vectors = motion_vectors_to_image(motion_vectors, (frame.shape[1], frame.shape[0]))
            motion_vectors = torch.from_numpy(motion_vectors).float()

            # get ground truth boxes andudate previous boxes and ids
            gt_boxes = self.gt_boxes_all[self.current_frame_idx]
            gt_ids = self.gt_ids_all[self.current_frame_idx]
            gt_boxes_prev_ = np.copy(self.gt_boxes_prev)
            gt_ids_prev_ = np.copy(self.gt_ids_prev)
            self.gt_boxes_prev = gt_boxes
            self.gt_ids_prev = gt_ids

            # match ids with previous ids and compute box velocities
            _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev_, assume_unique=True, return_indices=True)
            boxes = torch.from_numpy(gt_boxes[idx_1]).float()
            boxes_prev = torch.from_numpy(gt_boxes_prev_[idx_0]).float()
            velocities = velocities_from_boxes(boxes_prev, boxes)
            if self.visu:
                gt_ids = torch.from_numpy(gt_ids[idx_1])
                gt_ids_prev_ = torch.from_numpy(gt_ids_prev_[idx_0])

            # insert frame index into boxes and boxes_prev
            num_boxes = (boxes.shape)[0]
            boxes_prev_tmp = torch.zeros(num_boxes, 5).float()
            boxes_prev_tmp[:, 1:5] = boxes_prev
            boxes_prev_tmp[:, 0] = torch.full((num_boxes,), self.current_frame_idx).float()
            boxes_prev = boxes_prev_tmp
            if self.visu:
                boxes_tmp = torch.zeros(num_boxes, 5).float()
                boxes_tmp[:, 1:5] = boxes
                boxes_tmp[:, 0] = torch.full((num_boxes,), self.current_frame_idx).float()
                boxes = boxes_tmp

            # pad boxes_prev to the same global length (for MOT17 this is 52)
            boxes_prev_padded = torch.zeros(self.pad_num_boxes, 5).float()
            boxes_prev_padded[:num_boxes, :] = boxes_prev
            boxes_prev = boxes_prev_padded

            # similarly pad velocites
            velocities_padded = torch.zeros(self.pad_num_boxes, 4).float()
            velocities_padded[:num_boxes, :] = velocities
            velocities = velocities_padded

            if self.visu:
                # similarly pad boxes
                boxes_padded = torch.zeros(self.pad_num_boxes, 5).float()
                boxes_padded[:num_boxes, :] = boxes
                boxes = boxes_padded

                # similarly pad gt_ids
                gt_ids_padded = torch.zeros(self.pad_num_boxes,)
                gt_ids_padded[:num_boxes] = gt_ids
                gt_ids = gt_ids_padded
                gt_ids_prev_padded = torch.zeros(self.pad_num_boxes,)
                gt_ids_prev_padded[:num_boxes] = gt_ids_prev_
                gt_ids_prev_ = gt_ids_prev_padded

            # create a mask to revert the padding at a later stage
            num_boxes_mask = torch.zeros(self.pad_num_boxes,).bool()
            num_boxes_mask[0:num_boxes] = torch.ones(num_boxes,).bool()

            self.current_frame_idx += 1

            if self.visu:
                return (frame, frame_type, motion_vectors, boxes_prev, velocities, num_boxes_mask,
                    boxes, gt_ids, gt_ids_prev_)

            return motion_vectors, boxes_prev, velocities, num_boxes_mask


# run as python -m lib.datasets from root dir
if __name__ == "__main__":
    batch_size = 2
    datasets = {x: MotionVectorDataset(root_dir='../benchmark', visu=True, mode=x) for x in ["train", "val", "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=False, num_workers=8) for x in ["train", "val"]}

    step_wise = False

    for batch_idx in range(batch_size):
        cv2.namedWindow("frame-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame-{}".format(batch_idx), 640, 360)
    cv2.namedWindow("motion_vectors", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("motion_vectors", 640, 360)

    for step, (frames_, frame_tyes_, motion_vectors_, boxes_prev_, velocities_, num_boxes_mask_,
        boxes_, gt_ids_, gt_ids_prev_) in enumerate(dataloaders["train"]):

        for batch_idx in range(batch_size):

            frames = frames_[batch_idx]
            motion_vectors = motion_vectors_[batch_idx]
            boxes = boxes_[batch_idx]
            boxes_prev = boxes_prev_[batch_idx]
            gt_ids = gt_ids_[batch_idx]
            gt_ids_prev = gt_ids_prev_[batch_idx]
            velocities = velocities_[batch_idx]
            num_boxes_mask = num_boxes_mask_[batch_idx]

            boxes = boxes[:, 1:].numpy()
            boxes_prev = boxes_prev[:, 1:].numpy()

            frame = frames.numpy()
            motion_vectors = motion_vectors.numpy()
            gt_ids = gt_ids.long().numpy()
            gt_ids_prev = gt_ids_prev.long().numpy()

            #frame = draw_motion_vectors(frame, motion_vectors)
            frame = draw_boxes(frame, boxes, gt_ids, color=(255, 255, 255))
            frame = draw_boxes(frame, boxes_prev, gt_ids_prev, color=(200, 200, 200))
            frame = draw_velocities(frame, boxes, velocities)

            cv2.imshow("frame-{}".format(batch_idx), frame)
            cv2.imshow("motion_vectors", motion_vectors)

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
