import numpy as np
import cv2
import torch


def standardize(motion_vectors, mean, std):
    """Subtracts mean from motion vectors and divides by standard deviation.

    motion_vectors[channel] = (motion_vectors[channel] - mean[channel]) / std[channel]

    Args:
        motion_vectors (`torch.Tensor`): Motion vector image with shape (B x H x W X C)
            and channel order BGR. Blue channel has no meaning, green channel
            corresponds to y component of motion vector and red channel to x
            component of motion vector.

        mean (`list` of `float`): Mean values for blue, green and red channel to
            be subtracted from the motion vector image.

        std (`list` of `float`): Standard deviations per channel by which to
            divide the mean subtracted motion vector image.

    Returns:
        (`torch.Tensor`) the standardized motion vector image with same shape
        as input.
    """
    motion_vectors = (motion_vectors - torch.tensor(mean)) / torch.tensor(std)
    return motion_vectors


def scale_image(frame, short_side_min_len=600, long_side_max_len=1000):
    """Scale the input frame to match minimum and maximum size requirements.

    Frame is scaled so that it's shortest side has the size specified by the
    `short_side_min_len` parameter. The aspect ratio is constant. In case, the
    longer side would exceed the value specified by `long_side_max_len` a new
    scaling factor is computed so that the longer side matches `long_side_max_len`.
    The shorter side is then smaller than specified by `short_side_min_len`.

    Args:
        frame (`torch.Tensor`): Batch of frames with shape (B, H, W, C) with
            C = 3 and arbitrary datatype. Alternatively a single frame of shape
            (H, W, C) can be provided. The frames are resized based on the
            provided scaling factors.

        short_side_min_len (`int` or `float`): The desired size of the shorter
            side of the frame after scaling.

        long_side_max_len (`int` or `float`): The desired maximum size of the
            longer side of the frame after scaling.

    Returns:
        (`tuple`): First item is the batch of resized frames (`torch.Tensor`) or
        the single resized frame depending of the input. Second item is the
        computed scaling factor (`float`) which was used to scale both height
        and width of the input frames.
    """
    # determine the scaling factor
    frame = frame.numpy()
    frame_size_min = np.min(frame.shape[-3:-1])
    frame_size_max = np.max(frame.shape[-3:-1])
    scaling_factor = float(short_side_min_len) / float(frame_size_min)
    if np.round(scaling_factor * frame_size_max) > long_side_max_len:
        scaling_factor = float(long_side_max_len) / float(frame_size_max)
    # scale the frame
    if frame.ndim == 3:
        frame_resized = cv2.resize(frame, None, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    elif frame.ndim == 4:
        frame_resized = []
        for batch_idx in range(frame.shape[0]):
            frame_resized.append(cv2.resize(frame[batch_idx, ...], None, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR))
        frame_resized = np.stack(frame_resized, axis=0)
    else:
        raise ValueError("Invalid frame dimension")
    return torch.from_numpy(frame_resized), scaling_factor


# TESTING
if __name__ == "__main__":

    #####################
    ##   scale_image   ##
    #####################

    # single image
    image = torch.zeros((1080, 1920, 3))
    image_resized, scaling_factor = scale_image(image, short_side_min_len=600, long_side_max_len=1000)
    print(image_resized.shape, scaling_factor)
    assert image_resized.shape == (562, 1000, 3)
    assert int(np.round(scaling_factor*10000)) == 5208

    # single image different scale
    image = torch.zeros((1080, 1920, 3))
    image_resized, scaling_factor = scale_image(image, short_side_min_len=480, long_side_max_len=1000)
    print(image_resized.shape, scaling_factor)
    assert image_resized.shape == (480, 853, 3)
    assert int(np.round(scaling_factor*10000)) == 4444

    # batch of images
    image = torch.zeros((2, 1080, 1920, 3))
    image_resized, scaling_factor = scale_image(image, short_side_min_len=600, long_side_max_len=1000)
    print(image_resized.shape, scaling_factor)
    assert image_resized.shape == (2, 562, 1000, 3)
    assert int(np.round(scaling_factor*10000)) == 5208
