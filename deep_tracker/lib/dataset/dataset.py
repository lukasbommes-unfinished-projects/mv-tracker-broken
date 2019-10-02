import os
import glob
import pickle
import torch


class MotionVectorDatasetPrecomputed(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.items = ["motion_vectors", "boxes_prev", "velocities", "num_boxes_mask"]
        self.dirs = {}
        for item in self.items:
            self.dirs[item] = os.path.join(root_dir, item)
        # get dataset length
        self.length = len(glob.glob(os.path.join(self.dirs[self.items[0]], "*.pkl")))
        print(self.length)

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


# run as python -m lib.dataset.dataset from root dir
if __name__ == "__main__":

    root_dir = "precompute_dataset/data/"
    modes = ["train", "val"]
    datasets = {x: MotionVectorDatasetPrecomputed(root_dir=os.path.join(root_dir, x)) for x in modes}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1, shuffle=False, num_workers=0) for x in modes}

    for mode in modes:

        for step, (motion_vectors, boxes_prev, velocities, num_boxes_mask) in enumerate(dataloaders[mode]):

            print("Step: {}".format(step))

            print(motion_vectors.shape)
            print(boxes_prev.shape)
            print(velocities.shape)
            print(num_boxes_mask.shape)
