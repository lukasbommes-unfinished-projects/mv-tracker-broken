import torch
from torch.nn.parameter import Parameter


def load_pretrained_weights_to_modified_resnet(cnn_model, pretrained_weights):
    pre_dict = cnn_model.state_dict()
    for key, val in pretrained_weights.items():
        if key[0:5] == 'layer':
            key_list = key.split('.')
            tmp = int(int(key_list[1]) * 2)
            key_list[1] = str(tmp)
            tmp_key = ''
            for i in range(len(key_list)):
                tmp_key = tmp_key + key_list[i] + '.'
            key = tmp_key[:-1]
        if isinstance(val, Parameter):
            val = val.data
        pre_dict[key].copy_(val)
    cnn_model.load_state_dict(pre_dict)


def scale_image(frame, short_side_min_len=600, long_side_max_len=1000):
    """Scale the input frame to match minimum and maximum size requirements.

    Frame is scaled so that it's shortest side has the size specified by the
    `short_side_min_len` parameter. The aspect ratio is constant. In case, the
    longer side would exceed the value specified by `long_side_max_len` a new
    scaling factor is computed so that the longer side matches `long_side_max_len`.
    The shorter side is then smaller than specified by `short_side_min_len`.

    Args:
        frame (`numpy.ndarray`): Frame of shape (H, W, C) with C = 3 and arbitrary
            datatype. This frame is resized based on the provided scaling factors.

        short_side_min_len (`int` or `float`): The desired size of the shorter
            side of the frame after scaling.

        long_side_max_len (`int` or `float`): The desired maximum size of the
            longer side of the frame after scaling.

    Returns:
        (`tuple`): First item is the resized frame (`numpy.ndarray`). Second
        item is the computed scaling factor (`float`) which was used to scale
        both height and width of the input frame.
    """
    # determine the scaling factor
    frame_size_min = np.min(frame.shape[0:2])
    frame_size_max = np.max(frame.shape[0:2])
    scaling_factor = float(short_side_min_len) / float(frame_size_min)
    if np.round(scaling_factor * frame_size_max) > long_side_max_len:
        scaling_factor = float(long_side_max_len) / float(frame_size_max)
    # scale the frame
    print(scaling_factor)
    frame = cv2.resize(frame, None, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    return frame, scaling_factor
