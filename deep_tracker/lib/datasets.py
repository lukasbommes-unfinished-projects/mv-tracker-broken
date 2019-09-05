import os
import pickle
import torch
import cv2
import numpy as np

from lib.utils import velocities_from_boxes, box_from_velocities
from lib.visu import draw_boxes, draw_motion_vectors, draw_velocities


# class MotionVectorDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, mode, keyframe_interval=10, pad_num_boxes=52, window_length=3):
#         self.mode = mode
#         self.keyframe_interval = keyframe_interval
#         self.pad_num_boxes = pad_num_boxes
#         self.window_length = window_length
#         assert self.window_length > 0, "window length must be 1 or greater"
#         data_file = os.path.join(root_dir, "preprocessed", mode, "data.pkl")
#         self.data = pickle.load(open(data_file, "rb"))
#
#         # remove entries so that an integer number of sequences of window length can be generated
#         if self.window_length > 1:
#             lengths_file = os.path.join(root_dir, "preprocessed", mode, "lengths.pkl")
#             lengths = pickle.load(open(lengths_file, "rb"))
#             lenghts_cumsum = np.cumsum(lengths)
#             for lenght, lenght_cumsum in zip(reversed(lengths), reversed(lenghts_cumsum)):
#                 remainder = lenght % self.window_length  # e.g. 2
#                 for r in range(remainder):
#                     self.data.pop(lenght_cumsum-1-r)
#
#
#     def __len__(self):
#         return len(self.data) - 1  # -1 is needed because of idx+1 in __getitem__
#
#
#     def __getitem__(self, idx):
#
#         if self.mode == "train" or self.mode == "val":
#
#             motion_vectors_agg = []
#             boxes_prev_agg = []
#             velocities_agg = []
#             num_boxes_mask_agg = []
#
#             # for every sample we expect 3 (window_length) entries
#             for ws in range(self.window_length):  # ws = 0, 1, 2
#
#                 motion_vectors = self.data[idx + ws + 1]["motion_vectors"]
#
#                 frame_idx = self.data[idx + ws + 1]["frame_idx"]
#                 gt_ids = self.data[idx + ws + 1]["gt_ids"]
#                 gt_boxes = self.data[idx + ws + 1]["gt_boxes"]
#                 gt_ids_prev = self.data[idx + ws]["gt_ids"]
#                 gt_boxes_prev = self.data[idx + ws]["gt_boxes"]
#
#                 # find boxes which occured in the last frame
#                 _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev, assume_unique=True, return_indices=True)
#                 boxes = torch.from_numpy(gt_boxes[idx_1])
#                 boxes_prev = torch.from_numpy(gt_boxes_prev[idx_0])
#                 velocities = velocities_from_boxes(boxes_prev, boxes)
#
#                 # insert frame index into boxes
#                 num_boxes = (boxes.shape)[0]
#                 boxes_prev_tmp = torch.zeros(num_boxes, 5)
#                 boxes_prev_tmp[:, 1:5] = boxes_prev
#                 boxes_prev_tmp[:, 0] = torch.full((num_boxes,), frame_idx)
#                 boxes_prev = boxes_prev_tmp
#
#                 # pad boxes to the same global length (for MOT17 this is 52)
#                 boxes_prev_padded = torch.zeros(self.pad_num_boxes, 5)
#                 boxes_prev_padded[:num_boxes, :] = boxes_prev
#                 boxes_prev = boxes_prev_padded
#
#                 # similarly pad velocites
#                 velocities_padded = torch.zeros(self.pad_num_boxes, 4)
#                 velocities_padded[:num_boxes, :] = velocities
#                 velocities = velocities_padded
#
#                 # create a mask to revert the padding at a later stage
#                 num_boxes_mask = torch.zeros(self.pad_num_boxes,)
#                 num_boxes_mask[0:num_boxes] = torch.ones(num_boxes,)
#
#                 motion_vectors_agg.append(motion_vectors.float())
#                 boxes_prev_agg.append(boxes_prev.float())
#                 velocities_agg.append(velocities.float())
#                 num_boxes_mask_agg.append(num_boxes_mask.bool())
#
#             # convert aggregate lists to torch tensors
#             if self.window_length > 1:
#                 motion_vectors_agg = [t.unsqueeze(0) for t in motion_vectors_agg]
#                 boxes_prev_agg = [t.unsqueeze(0) for t in boxes_prev_agg]
#                 velocities_agg = [t.unsqueeze(0) for t in velocities_agg]
#                 num_boxes_mask_agg = [t.unsqueeze(0) for t in num_boxes_mask_agg]
#
#             motion_vectors_ret = torch.cat(motion_vectors_agg, axis=0)
#             boxes_prev_ret = torch.cat(boxes_prev_agg, axis=0)
#             velocities_ret = torch.cat(velocities_agg, axis=0)
#             num_boxes_mask_ret = torch.cat(num_boxes_mask_agg, axis=0)
#
#             # TODO: remove I frames
#
#             return motion_vectors_ret, boxes_prev_ret, velocities_ret, num_boxes_mask_ret


class MotionVectorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, codec, mode, pad_num_boxes=52, window_length=1, visu=False):
        self.root_dir = root_dir
        self.mode = mode
        self.pad_num_boxes = pad_num_boxes
        self.window_length = window_length
        self.visu = visu
        assert self.window_length > 0, "window length must be 1 or greater"
        data_file = os.path.join(root_dir, "preprocessed", codec, mode, "data.pkl")
        self.data = pickle.load(open(data_file, "rb"))

        # remove entries so that an integer number of sequences of window length can be generated
        if self.window_length > 1:
            lengths_file = os.path.join(root_dir, "preprocessed", codec, mode, "lengths.pkl")
            lengths = pickle.load(open(lengths_file, "rb"))
            lenghts_cumsum = np.cumsum(lengths)
            for lenght, lenght_cumsum in zip(reversed(lengths), reversed(lenghts_cumsum)):
                remainder = lenght % self.window_length  # e.g. 2
                for r in range(remainder):
                    self.data.pop(lenght_cumsum-1-r)


    def __len__(self):
        return len(self.data) - 1  # -1 is needed because of idx+1 in __getitem__


    def __getitem__(self, idx):

        if self.mode == "train" or self.mode == "val":

            motion_vectors_agg = []
            boxes_prev_agg = []
            velocities_agg = []
            num_boxes_mask_agg = []
            if self.visu:
                frame_agg = []
                boxes_agg = []
                gt_ids_agg = []
                gt_ids_prev_agg = []

            # for every sample we expect 3 (window_length) entries
            for ws in range(self.window_length):  # ws = 0, 1, 2

                frame_idx = self.data[idx + ws + 1]["frame_idx"]

                motion_vectors = self.data[idx + ws + 1]["motion_vectors"]

                # get current frame
                if self.visu:
                    sequence = self.data[idx + ws + 1]["sequence"]
                    frame_file = os.path.join(self.root_dir, "train", "{}-FRCNN/img1/{:06d}.jpg".format(sequence, frame_idx+1))
                    frame = torch.from_numpy(cv2.imread(frame_file, cv2.IMREAD_COLOR))

                gt_ids = self.data[idx + ws + 1]["gt_ids"]
                gt_boxes = self.data[idx + ws + 1]["gt_boxes"]
                gt_ids_prev = self.data[idx + ws]["gt_ids"]
                gt_boxes_prev = self.data[idx + ws]["gt_boxes"]

                # find boxes which occured in the last frame
                _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev, assume_unique=True, return_indices=True)
                boxes = torch.from_numpy(gt_boxes[idx_1])
                boxes_prev = torch.from_numpy(gt_boxes_prev[idx_0])
                velocities = velocities_from_boxes(boxes_prev, boxes)
                if self.visu:
                    gt_ids = torch.from_numpy(gt_ids[idx_1])
                    gt_ids_prev = torch.from_numpy(gt_ids_prev[idx_0])

                # scale boxes to match the motion vector dimensions
                # factor of 16 corresponds to 16 x 16 macroblocks
                boxes = boxes / 16.0
                boxes_prev = boxes_prev / 16.0


                # insert frame index into boxes and boxes_prev
                num_boxes = (boxes.shape)[0]
                boxes_prev_tmp = torch.zeros(num_boxes, 5)
                boxes_prev_tmp[:, 1:5] = boxes_prev
                boxes_prev_tmp[:, 0] = torch.full((num_boxes,), frame_idx)
                boxes_prev = boxes_prev_tmp
                if self.visu:
                    boxes_tmp = torch.zeros(num_boxes, 5)
                    boxes_tmp[:, 1:5] = boxes
                    boxes_tmp[:, 0] = torch.full((num_boxes,), frame_idx)
                    boxes = boxes_tmp

                # pad boxes_prev to the same global length (for MOT17 this is 52)
                boxes_prev_padded = torch.zeros(self.pad_num_boxes, 5)
                boxes_prev_padded[:num_boxes, :] = boxes_prev
                boxes_prev = boxes_prev_padded

                # similarly pad velocites
                velocities_padded = torch.zeros(self.pad_num_boxes, 4)
                velocities_padded[:num_boxes, :] = velocities
                velocities = velocities_padded

                if self.visu:
                    # similarly pad boxes
                    boxes_padded = torch.zeros(self.pad_num_boxes, 5)
                    boxes_padded[:num_boxes, :] = boxes
                    boxes = boxes_padded

                    # similarly pad gt_ids
                    gt_ids_padded = torch.zeros(self.pad_num_boxes,)
                    gt_ids_padded[:num_boxes] = gt_ids
                    gt_ids = gt_ids_padded
                    gt_ids_prev_padded = torch.zeros(self.pad_num_boxes,)
                    gt_ids_prev_padded[:num_boxes] = gt_ids_prev
                    gt_ids_prev = gt_ids_prev_padded

                # create a mask to revert the padding at a later stage
                num_boxes_mask = torch.zeros(self.pad_num_boxes,)
                num_boxes_mask[0:num_boxes] = torch.ones(num_boxes,)

                motion_vectors_agg.append(motion_vectors.float())
                boxes_prev_agg.append(boxes_prev.float())
                velocities_agg.append(velocities.float())
                num_boxes_mask_agg.append(num_boxes_mask.bool())
                if self.visu:
                    frame_agg.append(frame)
                    boxes_agg.append(boxes.float())
                    gt_ids_agg.append(gt_ids)
                    gt_ids_prev_agg.append(gt_ids_prev)

            # convert aggregate lists to torch tensors
            if self.window_length > 1:
                motion_vectors_agg = [t.unsqueeze(0) for t in motion_vectors_agg]
                boxes_prev_agg = [t.unsqueeze(0) for t in boxes_prev_agg]
                velocities_agg = [t.unsqueeze(0) for t in velocities_agg]
                num_boxes_mask_agg = [t.unsqueeze(0) for t in num_boxes_mask_agg]
                if self.visu:
                    frame_agg = [t.unsqueeze(0) for t in frame_agg]
                    boxes_agg = [t.unsqueeze(0) for t in boxes_agg]
                    gt_ids_agg = [t.unsqueeze(0) for t in gt_ids_agg]
                    gt_ids_prev_agg = [t.unsqueeze(0) for t in gt_ids_prev_agg]


            motion_vectors_ret = torch.cat(motion_vectors_agg, axis=0)
            boxes_prev_ret = torch.cat(boxes_prev_agg, axis=0)
            velocities_ret = torch.cat(velocities_agg, axis=0)
            num_boxes_mask_ret = torch.cat(num_boxes_mask_agg, axis=0)
            if self.visu:
                frame_ret = torch.cat(frame_agg, axis=0)
                boxes_ret = torch.cat(boxes_agg, axis=0)
                gt_ids_ret = torch.cat(gt_ids_agg, axis=0)
                gt_ids_prev_ret = torch.cat(gt_ids_prev_agg, axis=0)

                return motion_vectors_ret, boxes_prev_ret, velocities_ret, num_boxes_mask_ret, boxes_ret, gt_ids_ret, gt_ids_prev_ret, frame_ret

            return motion_vectors_ret, boxes_prev_ret, velocities_ret, num_boxes_mask_ret


# run as python -m lib.datasets from root dir
if __name__ == "__main__":
    batch_size = 2
    window_lenght = 1
    datasets = {x: MotionVectorDataset(root_dir='data', window_length=window_lenght, codec="mpeg4", visu=True, mode=x) for x in ["train", "val", "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ["train", "val", "test"]}

    step_wise = False

    for batch_idx in range(batch_size):
        cv2.namedWindow("frame-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame-{}".format(batch_idx), 640, 360)

    for step, (motion_vectors_, boxes_prev_, velocities_, num_boxes_mask_, boxes_,
        gt_ids_, gt_ids_prev_, frames_) in enumerate(dataloaders["train"]):

        for batch_idx in range(batch_size):

            if window_lenght == 1:

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
                gt_ids = gt_ids.long().numpy()
                gt_ids_prev = gt_ids_prev.long().numpy()

                frame = draw_motion_vectors(frame, motion_vectors)
                frame = draw_boxes(frame, boxes*16.0, gt_ids, color=(255, 255, 255))
                frame = draw_boxes(frame, boxes_prev*16.0, gt_ids_prev, color=(200, 200, 200))
                frame = draw_velocities(frame, boxes, velocities)

                cv2.imshow("frame-{}".format(batch_idx), frame)

            elif window_lenght > 1:
                raise NotImplementedError("Visualization of data with window length > 1 is not yet supported.")

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
