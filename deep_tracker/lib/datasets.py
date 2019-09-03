import os
import pickle
import torch
import numpy as np

from lib.utils import velocities_from_boxes, box_from_velocities


class MotionVectorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, keyframe_interval=10, pad_num_boxes=52, window_length=3):
        self.mode = mode
        self.keyframe_interval = keyframe_interval
        self.pad_num_boxes = pad_num_boxes
        self.window_length = window_length
        assert self.window_length > 0, "window length must be 1 or greater"
        data_file = os.path.join(root_dir, "preprocessed", mode, "data.pkl")
        self.data = pickle.load(open(data_file, "rb"))

        # remove entries so that an integer number of sequences of window length can be generated
        if self.window_length > 1:
            lengths_file = os.path.join(root_dir, "preprocessed", mode, "lengths.pkl")
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

            # for every sample we expect 3 (window_length) entries
            for ws in range(self.window_length):  # ws = 0, 1, 2

                motion_vectors = self.data[idx + ws + 1]["motion_vectors"]

                frame_idx = self.data[idx + ws + 1]["frame_idx"]
                gt_ids = self.data[idx + ws + 1]["gt_ids"]
                gt_boxes = self.data[idx + ws + 1]["gt_boxes"]
                gt_ids_prev = self.data[idx + ws]["gt_ids"]
                gt_boxes_prev = self.data[idx + ws]["gt_boxes"]

                # find boxes which occured in the last frame
                _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev, assume_unique=True, return_indices=True)
                boxes = torch.from_numpy(gt_boxes[idx_1])
                boxes_prev = torch.from_numpy(gt_boxes_prev[idx_0])
                velocities = velocities_from_boxes(boxes_prev, boxes)

                # insert frame index into boxes
                num_boxes = (boxes.shape)[0]
                boxes_prev_tmp = torch.zeros(num_boxes, 5)
                boxes_prev_tmp[:, 1:5] = boxes_prev
                boxes_prev_tmp[:, 0] = torch.full((num_boxes,), frame_idx)
                boxes_prev = boxes_prev_tmp

                # pad boxes to the same global length (for MOT17 this is 52)
                boxes_prev_padded = torch.zeros(self.pad_num_boxes, 5)
                boxes_prev_padded[:num_boxes, :] = boxes_prev
                boxes_prev = boxes_prev_padded

                # similarly pad velocites
                velocities_padded = torch.zeros(self.pad_num_boxes, 4)
                velocities_padded[:num_boxes, :] = velocities
                velocities = velocities_padded

                # create a mask to revert the padding at a later stage
                num_boxes_mask = torch.zeros(self.pad_num_boxes,)
                num_boxes_mask[0:num_boxes] = torch.ones(num_boxes,)

                motion_vectors_agg.append(motion_vectors.float())
                boxes_prev_agg.append(boxes_prev.float())
                velocities_agg.append(velocities.float())
                num_boxes_mask_agg.append(num_boxes_mask.bool())

            # convert aggregate lists to torch tensors
            if self.window_length > 1:
                motion_vectors_agg = [t.unsqueeze(0) for t in motion_vectors_agg]
                boxes_prev_agg = [t.unsqueeze(0) for t in boxes_prev_agg]
                velocities_agg = [t.unsqueeze(0) for t in velocities_agg]
                num_boxes_mask_agg = [t.unsqueeze(0) for t in num_boxes_mask_agg]

            motion_vectors_ret = torch.cat(motion_vectors_agg, axis=0)
            boxes_prev_ret = torch.cat(boxes_prev_agg, axis=0)
            velocities_ret = torch.cat(velocities_agg, axis=0)
            num_boxes_mask_ret = torch.cat(num_boxes_mask_agg, axis=0)

            # TODO: remove I frames

            return motion_vectors_ret, boxes_prev_ret, velocities_ret, num_boxes_mask_ret


if __name__ == "__main__":
    datasets = {x: MotionVectorDataset(root_dir='../data', window_length=7, mode=x) for x in ["train", "val", "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=7, shuffle=False, num_workers=4) for x in ["train", "val", "test"]}

    for step, (motion_vectors, boxes_prev, velocities, num_boxes_mask) in enumerate(dataloaders["train"]):
        print(step)
        print(motion_vectors.shape)
        print(boxes_prev.shape)
        print(velocities.shape)
        print(num_boxes_mask.shape)
