import os
import glob
import pickle
import torch

import cv2


class MotionVectorDatasetPrecomputed(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.items = ["motion_vectors", "boxes_prev", "velocities", "num_boxes_mask", "motion_vector_scale"]
        self.dirs = {}
        for item in self.items:
            self.dirs[item] = os.path.join(root_dir, item)
        # get dataset length
        self.length = len(glob.glob(os.path.join(self.dirs[self.items[0]], "*.pkl")))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError
        data = {}
        for item in self.items:
            file = os.path.join(self.dirs[item], "{:08d}.pkl".format(idx))
            data[item] = pickle.load(open(file, "rb"))
        return tuple(data.values())


# run as python -m lib.dataset.dataset_precomputed from root dir
if __name__ == "__main__":

    root_dir = "data_precomputed"
    modes = ["train", "val"]
    datasets = {x: MotionVectorDatasetPrecomputed(root_dir=os.path.join(root_dir, x)) for x in modes}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1, shuffle=False, num_workers=8) for x in modes}

    for mode in modes:

        step_wise = False

        for batch_idx in range(2):
            cv2.namedWindow("motion_vectors-{}".format(batch_idx), cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_vectors-{}".format(batch_idx), 640, 360)

        for step, (motion_vectors, boxes_prev, velocities,
            num_boxes_mask, motion_vector_scale) in enumerate(dataloaders[mode]):

            # remove batch dimension as precomputed data is already batched
            motion_vectors.squeeze_(0)
            boxes_prev.squeeze_(0)
            velocities.squeeze_(0)
            num_boxes_mask.squeeze_(0)
            motion_vector_scale.squeeze_(0)

            print("Step: {}".format(step))

            print(motion_vectors.shape)
            print(boxes_prev.shape)
            print(velocities.shape)
            print(num_boxes_mask.shape)
            print(motion_vector_scale)

            for batch_idx in range(motion_vectors.shape[0]):

                motion_vectors_ = motion_vectors[batch_idx]
                boxes_prev_ = boxes_prev[batch_idx]
                velocities_ = velocities[batch_idx]
                num_boxes_mask_ = num_boxes_mask[batch_idx]

                print(type(motion_vectors_))
                print(motion_vectors_.shape)
                motion_vectors_ = motion_vectors_[[2, 1, 0], ...]
                motion_vectors_ = motion_vectors_.permute(1, 2, 0)
                motion_vectors_ = motion_vectors_.numpy()
                print(type(motion_vectors_))
                print(motion_vectors_.shape)

                print("step: {}, MVS shape: {}".format(step, motion_vectors_.shape))
                cv2.imshow("motion_vectors-{}".format(batch_idx), motion_vectors_)

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
