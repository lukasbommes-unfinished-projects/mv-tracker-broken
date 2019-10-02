import math
import numpy as np
import torch


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


def motion_vectors_to_image(motion_vectors, frame_shape=(1920, 1080), scale=False):
    """Converts a set of motion vectors into a BGR image.

    Args:
        motion_vectors (`numpy.ndarray`): Motion vector array with shape [N, 10]
            as returned by VideoCap. The motion vector array should only contain P-vectors
            which can be filtered out by using get_vectors_by_source(motion_vectors, "past").
            Also, the reference frame should be normalized by using normalize_vectors.

        frame_shape (`tuple` of `int`): Desired (width, height) in pixels of the returned image.
            Should correspond to the size of the source footage of which the motion vectors
            where extracted.

        scale (`bool`): If True, scale motion vector components in the output image to
            range [0, 1]. If False, do not scale.

    Returns:
        `numpy.ndarray` The motion vectors encoded as image. Image shape is (height, widht, 3)
        and channel order is BGR. The red channel contains the x motion components of
        the motion vectors and the green channel the y motion components.
    """
    # compute necessary frame shape
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

        if scale:
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
