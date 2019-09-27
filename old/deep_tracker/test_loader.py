#from lib.dataset.loaders import load_groundtruth
import numpy as np


def load_groundtruth(gt_file, num_frames, only_eval=False):
    """
    Args:
        gt_file (`string`): Full path of a MOT groundtruth txt file.

        num_frames (`int`): The number of frames in the corresponding MOT sequence.

        only_eval (`bool`): If False load all groundtruth entries, otherwise
            load only entries in which column 7 is 1 indicating an entry that
            is to be considered during evaluation.
    """
    gt_boxes = []
    gt_ids = []
    gt_classes = []
    raw_data = np.genfromtxt(gt_file, delimiter=',')
    print(raw_data)

    for frame_idx in range(1, num_frames + 1):
        idx = np.where(raw_data[:, 0] == frame_idx)
        if idx[0].size:  # if there are boxes in this frame
            gt_box = np.stack(raw_data[idx], axis=0)[:, 2:6]
            gt_id = np.stack(raw_data[idx], axis=0)[:, 1]
            gt_class = np.stack(raw_data[idx], axis=0)[:, 7]
            consider_in_eval = np.stack(raw_data[idx], axis=0)[:, 6]
            consider_in_eval = consider_in_eval.astype(np.bool)
            if only_eval:
                gt_box = gt_box[consider_in_eval]
                gt_id = gt_id[consider_in_eval]
            if len(gt_id) > 0:
                gt_boxes.append(gt_box)
                gt_ids.append(gt_id)
                gt_classes.append(gt_class)
            else:
                gt_boxes.append(None)
                gt_ids.append(None)
                gt_classes.append(None)
        else:
            gt_boxes.append(None)
            gt_ids.append(None)
            gt_classes.append(None)
    return gt_ids, gt_boxes, gt_classes


gt_file = "data/MOT15/train/KITTI-17/gt/gt.txt"
#gt_file = "data/MOT15/train/ADL-Rundle-6/gt/gt.txt"
gt_ids, gt_boxes, _ = load_groundtruth(gt_file, num_frames=145, only_eval=True)

print("len gt_boxes:", len(gt_boxes))
print("len gt_ids:", len(gt_ids))

print(gt_ids[1])
