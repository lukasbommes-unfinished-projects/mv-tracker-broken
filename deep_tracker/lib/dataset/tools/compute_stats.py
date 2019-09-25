import cv2
import numpy as np
import torch

from lib.dataset.dataset import MotionVectorDataset


class RunningStats():
    def __init__(self):
        self.existingAggregate = (0, 0, 0)

    def update(self, motion_vectors_channel):
        newValue = np.mean(motion_vectors_channel)
        (count, mean, M2) = self.existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        self.existingAggregate = (count, mean, M2)

    # retrieve the mean, variance and sample variance from an aggregate
    def get_stats(self):
        (count, mean, M2) = self.existingAggregate
        (mean, variance) = (mean, M2/count)
        if count < 2:
            return float('nan')
        else:
            return (mean, variance)


# run as python -m lib.dataset.tools.compute_stats from root dir
if __name__ == "__main__":
    visu = False  # whether to show graphical output (frame + motion vectors) or not
    dataset_train = MotionVectorDataset(root_dir='data', batch_size=1, visu=visu, mode="train")
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

    step_wise = False

    if visu:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 640, 360)
        cv2.namedWindow("motion_vectors", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_vectors", 640, 360)

    runnings_stats_x = RunningStats()
    runnings_stats_y = RunningStats()

    for step, ret in enumerate(dataloader_train):

        if visu:
            (frames, motion_vectors, _, _, _) = ret
        else:
            (motion_vectors, _, _, _) = ret

        motion_vectors = motion_vectors[0].numpy()
        runnings_stats_x.update(motion_vectors[:, :, 2])
        runnings_stats_y.update(motion_vectors[:, :, 1])

        if visu:
            frame = frames[0].numpy()
            print("step: {}, MVS shape: {}".format(step, motion_vectors.shape))

            cv2.imshow("frame", frame)
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

    final_mean_x, final_var_x = runnings_stats_x.get_stats()
    final_std_x = np.sqrt(final_var_x)
    final_mean_y, final_var_y = runnings_stats_y.get_stats()
    final_std_y = np.sqrt(final_var_y)
    print("x_channel -- mean: {}, variance: {}, std: {}".format(final_mean_x, final_var_x, final_std_x))
    print("y_channel -- mean: {}, variance: {}, std: {}".format(final_mean_y, final_var_y, final_std_y))
