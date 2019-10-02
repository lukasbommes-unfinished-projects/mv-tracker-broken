# This script precomputes an offline dataset for faster training
import os
import errno
import pickle
from tqdm import tqdm
import torch

from lib.dataset.dataset import MotionVectorDataset
from lib.transforms.transforms import standardize, scale_image


# run as python -m lib.dataset.tools.precompute_dataset from root dir
if __name__ == "__main__":

    # configure desired dataset settings here
    batch_size = 2
    codec = "mpeg4"
    modes = ["train", "val"]  # which datasets to generate
    input_folder = "data"  # where to look for the input dataset, relative to root dir
    output_folder = "data_precomputed" # where to save the precomputed samples, relative to root dir

    items = ["motion_vectors", "boxes_prev", "velocities", "num_boxes_mask"]

    for mode in modes:

        for item in items:
            try:
                os.makedirs(os.path.join(output_folder, mode, item))
            except FileExistsError:
                msg = ("Looks like the output directory "
                f"'{output_folder}' already contains data. Manually move or "
                "delete this directory before proceeding.")
                raise FileExistsError(msg)

        dataset = MotionVectorDataset(root_dir=input_folder, batch_size=batch_size, codec=codec, pad_num_boxes=52, visu=False, mode=mode)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        print("Mode {} of {}".format(mode, modes))
        pbar = tqdm(total=len(dataloader))
        for step, (motion_vectors, boxes_prev, velocities, num_boxes_mask) in enumerate(dataloader):

            # standardize motion vectors
            if codec == "h264":
                motion_vectors = standardize(motion_vectors,
                    mean=[0.0, 0.3219420202390504, -0.3864056486553166],
                    std=[1.0, 1.277147814669969, 4.76270068707976])
            elif codec == "mpeg4":
                motion_vectors = standardize(motion_vectors,
                    mean=[0.0, 0.1770176594258104, -0.12560456383521534],
                    std=[1.0, 0.7420489598781672, 1.8279847980299613])

            # resize spatial dimsions of motion vectors
            motion_vectors, motion_vector_scale = scale_image(motion_vectors, short_side_min_len=600, long_side_max_len=1000)

            # swap channel order of motion vectors from BGR to RGB
            motion_vectors = motion_vectors[..., [2, 1, 0]]

            # swap motion vector axes so that shape is (B, C, H, W) instead of (B, H, W, C)
            motion_vectors = motion_vectors.permute(0, 3, 1, 2)

            data = {
                "motion_vectors": motion_vectors,
                "boxes_prev": boxes_prev,
                "velocities": velocities,
                "num_boxes_mask": num_boxes_mask
            }

            # save data into output folder
            for item in items:
                output_file = os.path.join(output_folder, mode, item, "{:08d}.pkl".format(step))
                pickle.dump(data[item], open(output_file, "wb"))

            pbar.update()

            if step == 100:
                break

        pbar.close()
