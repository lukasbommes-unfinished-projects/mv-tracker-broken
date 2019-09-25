import os
import torch
import cv2
import numpy as np

from video_cap import VideoCap

from lib.dataset.loaders import load_detections, load_groundtruth
from lib.dataset.motion_vectors import get_vectors_by_source, get_nonzero_vectors, \
    normalize_vectors, motion_vectors_to_image
from lib.dataset.velocities import velocities_from_boxes
from lib.visu import draw_boxes, draw_velocities


class MotionVectorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, batch_size, pad_num_boxes=52, visu=False):

        self.DEBUG = False  # whteher to print debug information

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
            "val": [525, 654],
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

        # pre-compute the last frame index at which to switch to the next sequence
        # so that the sequence length is dividable by the batch size. Otherwise, it
        # can happen that a batch is created that contains frames from two different
        # sequences.
        self.lens_truncated = []
        for seq_len in self.lens[self.mode]:
            self.lens_truncated.append(seq_len - ((seq_len - 1) % batch_size))

        print(self.lens_truncated)


    def __len__(self):
        total_len = sum(self.lens_truncated) - len(self.lens_truncated)  # first frame of each sequence is skipped
        if self.DEBUG:
            print("Overall length of dataset: {}".format(total_len))
        return total_len


    def __getitem__(self, idx):

        #TODO:
        # - load ground truth only for train and val data

        while True:

            if self.DEBUG:
                print("Getting item idx {}, Current sequence idx {}, Current frame idx {}".format(idx, self.current_seq_id, self.current_frame_idx))

            # when the end of the sequence is reached switch to the next one
            if self.current_frame_idx == self.lens_truncated[self.current_seq_id]:
                if self.DEBUG:
                    print("Sequence {} is being closed...".format(self.sequences[self.mode][self.current_seq_id]))
                self.caps[self.current_seq_id].release()
                self.is_open[self.current_seq_id] = False
                self.current_frame_idx = 0
                self.current_seq_id += 1
                # make sure the sequence index wraps around at the number of sequences
                if self.current_seq_id == len(self.sequences[self.mode]):
                    self.current_seq_id = 0
                if self.DEBUG:
                    print("Updated sequence id to {} and frame index to {}".format(self.current_seq_id, self.current_frame_idx))
                continue

            # this block is executed only once when a new sequence starts
            if not self.is_open[self.current_seq_id]:
                if self.DEBUG:
                    print("Sequence {} is being opened...".format(self.sequences[self.mode][self.current_seq_id]))

                # open detections and groundtruth files
                #detections_file = os.path.join(self.root_dir, self.sequences[self.current_seq_id], "det/det.txt")
                #detections = load_detections(detections_file, num_frames=self.lens[self.mode][self.current_seq_id])
                gt_file = os.path.join(self.root_dir, self.sequences[self.mode][self.current_seq_id], "gt/gt.txt")
                self.gt_ids_all, self.gt_boxes_all, _ = load_groundtruth(gt_file, only_eval=True)
                if self.DEBUG:
                    print("groundtruth files loaded")

                # read and store first set of ground truth boxes and ids
                self.gt_boxes_prev = self.gt_boxes_all[0]
                self.gt_ids_prev = self.gt_ids_all[0]

                # open the video sequence and drop first frame
                sequence_name = str.split(self.sequences[self.mode][self.current_seq_id], "/")[-1]
                video_file = os.path.join(self.root_dir, self.sequences[self.mode][self.current_seq_id], "{}-h264.mp4".format(sequence_name))
                if self.DEBUG:
                    print("Opening video file {} of sequence {}".format(video_file, sequence_name))
                ret = self.caps[self.current_seq_id].open(video_file)
                if not ret:
                    raise RuntimeError("Could not open the video file")
                _ = self.caps[self.current_seq_id].read()
                if self.DEBUG:
                    print("Opened the video file")

                self.is_open[self.current_seq_id] = True
                self.current_frame_idx += 1
                if self.DEBUG:
                    print("Incremented frame index to {}".format(self.current_frame_idx))
                continue

            # otherwise just load, process and return the next sample
            ret, frame, motion_vectors, frame_type, _ = self.caps[self.current_seq_id].read()
            if not ret:  # should never happen
                raise RuntimeError("Could not read next frame from video")
            if self.DEBUG:
                print("got frame, frame_type {}, mvs shape: {}, frame shape: {}".format(frame_type, motion_vectors.shape, frame.shape))

            # convert motion vectors to image (for I frame black image is returned)
            motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
            motion_vectors = normalize_vectors(motion_vectors)
            motion_vectors = get_nonzero_vectors(motion_vectors)
            motion_vectors = motion_vectors_to_image(motion_vectors, (frame.shape[1], frame.shape[0]))
            motion_vectors = torch.from_numpy(motion_vectors).float()

            # get ground truth boxes and update previous boxes and ids
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


# run as python -m lib.dataset.dataset from root dir
if __name__ == "__main__":
    batch_size = 1
    datasets = {x: MotionVectorDataset(root_dir='data', batch_size=batch_size, visu=True, mode=x) for x in ["train", "val", "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=False, num_workers=0) for x in ["train", "val"]}

    step_wise = False

    for batch_idx in range(batch_size):
        cv2.namedWindow("frame-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame-{}".format(batch_idx), 640, 360)
        cv2.namedWindow("motion_vectors-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_vectors-{}".format(batch_idx), 640, 360)

    for step, (frames_, frame_types_, motion_vectors_, boxes_prev_, velocities_, num_boxes_mask_,
        boxes_, gt_ids_, gt_ids_prev_) in enumerate(dataloaders["val"]):

        for batch_idx in range(batch_size):

            frames = frames_[batch_idx]
            frame_type = frame_types_[batch_idx]
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

            cv2.putText(frame, 'Frame Idx: {}'.format(step), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Frame Type: {}'.format(frame_type), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            #frame = draw_motion_vectors(frame, motion_vectors)
            frame = draw_boxes(frame, boxes, gt_ids, color=(255, 255, 255))
            frame = draw_boxes(frame, boxes_prev, gt_ids_prev, color=(200, 200, 200))
            frame = draw_velocities(frame, boxes, velocities)

            print("step: {}, MVS shape: {}".format(step, motion_vectors.shape))

            cv2.imshow("frame-{}".format(batch_idx), frame)
            cv2.imshow("motion_vectors-{}".format(batch_idx), motion_vectors)

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
