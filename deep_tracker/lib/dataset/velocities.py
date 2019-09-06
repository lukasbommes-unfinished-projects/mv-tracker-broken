import math
import torch
from torch.nn.parameter import Parameter


# def velocities_from_boxes(boxes_prev, boxes):
#     """Computes bounding box velocities.
#
#     Args:
#         boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [B, N, 4]
#             where B is the batch size and N the number of bounding boxes.
#
#         boxes (`torch.Tensor`): Bounding boxes in current frame. Shape [B, N, 4]
#             where B is the batch size and N the number of bounding boxes.
#
#     Returns:
#         (`torch.Tensor`) velocities of box coordinates in current frame. Shape [B, N, 4].
#
#     Ensure that ordering of boxes in both tensors is consistent and that the number of boxes
#     is the same.
#     """
#     x = boxes[:, 0]
#     y = boxes[:, 1]
#     w = boxes[:, 2]
#     h = boxes[:, 3]
#     x_p = boxes_prev[:, 0]
#     y_p = boxes_prev[:, 1]
#     w_p = boxes_prev[:, 2]
#     h_p = boxes_prev[:, 3]
#     v_x = (1 / w_p * (x - x_p)).unsqueeze(-1)
#     v_y = (1 / h_p * (y - y_p)).unsqueeze(-1)
#     v_w = (torch.log(w / w_p)).unsqueeze(-1)
#     v_h = (torch.log(h / h_p)).unsqueeze(-1)
#     return torch.cat([v_x, v_y, v_w, v_h], -1)
#
#
# def box_from_velocities(boxes_prev, velocities):
#     """Computes bounding boxes from previous boxes and velocities.
#
#     Args:
#         boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [B, N, 4]
#             where B is the batch size and N the number of bounding boxes.
#
#         velocities (`torch.Tensor`): Box velocities in current frame. Shape [B, N, 4]
#         where B is the batch size and N the number of bounding boxes.
#
#     Returns:
#         (`torch.Tensor`) Bounding boxes in current frame. Shape [B, N, 4].
#
#     Ensure that ordering of boxes and velocities in both tensors is consistent that is
#     box in row i should correspond to velocities in row i.
#     """
#     x_p = boxes_prev[:, 0]
#     y_p = boxes_prev[:, 1]
#     w_p = boxes_prev[:, 2]
#     h_p = boxes_prev[:, 3]
#     v_x = velocities[:, 0]
#     v_y = velocities[:, 1]
#     v_w = velocities[:, 2]
#     v_h = velocities[:, 3]
#     x = (w_p * v_x + x_p).unsqueeze(-1)
#     y = (h_p * v_y + y_p).unsqueeze(-1)
#     w = (w_p * torch.exp(v_w)).unsqueeze(-1)
#     h = (h_p * torch.exp(v_h)).unsqueeze(-1)
#     return torch.cat([x, y, w, h], -1)


def velocities_from_boxes(boxes_prev, boxes, sigma=1.5):
    """Computes bounding box velocities.

    Args:
        boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [B, N, 4]
            where B is the batch size and N the number of bounding boxes. Each row in
            this tensor corresponds to one bounding box in format [x, y, w, h] where
            x and y are the coordinates of the top left corner and w and h are box
            width and height.

        boxes (`torch.Tensor`): Bounding boxes in current frame. Shape [B, N, 4]
            where B is the batch size and N the number of bounding boxes. Same format
            as boxes_prev.

        sigma (`float`): A constant scaling factor which is multiplied with the box center velocities.

    Returns:
        (`torch.Tensor`) velocities of box coordinates in current frame. Shape [B, N, 4].
        Each row in this tensor corresponds to a box velocity in format [v_xc, v_yc, v_w, v_h]
        where v_xc and v_yc are velocities of the box center point and v_w and v_h are
        velocities of the box width and height.

    Ensure that ordering of boxes in both tensors is consistent and that the number of boxes
    is the same.
    """
    w = boxes[:, 2]
    h = boxes[:, 3]
    xc = boxes[:, 0] + 0.5 * w
    yc = boxes[:, 1] + 0.5 * h
    w_p = boxes_prev[:, 2]
    h_p = boxes_prev[:, 3]
    xc_p = boxes_prev[:, 0] + 0.5 * w_p
    yc_p = boxes_prev[:, 1] + 0.5 * h_p
    v_xc = (math.sqrt(2) * sigma / w_p * (xc - xc_p)).unsqueeze(-1)
    v_yc = (math.sqrt(2) * sigma / h_p * (yc - yc_p)).unsqueeze(-1)
    v_w = (torch.log(w / w_p)).unsqueeze(-1)
    v_h = (torch.log(h / h_p)).unsqueeze(-1)
    return torch.cat([v_xc, v_yc, v_w, v_h], -1)


def box_from_velocities(boxes_prev, velocities, sigma=1.5):
    """Computes bounding boxes from previous boxes and velocities.

    Args:
        boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [B, N, 4]
            where B is the batch size and N the number of bounding boxes.
            Each row in
            this tensor corresponds to one bounding box in format [x, y, w, h] where
            x and y are the coordinates of the top left corner and w and h are box
            width and height.

        velocities (`torch.Tensor`): Box velocities in current frame. Shape [B, N, 4]
            where B is the batch size and N the number of bounding boxes.
            Each row in this tensor corresponds to a box velocity in format [v_xc, v_yc, v_w, v_h]
            where v_xc and v_yc are velocities of the box center point and v_w and v_h are
            velocities of the box width and height.

        sigma (`float`): A constant scaling factor which is multiplied with the box center velocities.

    Returns:
        (`torch.Tensor`) Bounding boxes in current frame. Shape [B, N, 4]. Same format
        as boxes_prev.

    Ensure that ordering of boxes and velocities in both tensors is consistent that is
    box in row i should correspond to velocities in row i.
    """
    w_p = boxes_prev[:, 2]
    h_p = boxes_prev[:, 3]
    xc_p = boxes_prev[:, 0] + 0.5 * w_p
    yc_p = boxes_prev[:, 1] + 0.5 * h_p
    v_xc = velocities[:, 0]
    v_yc = velocities[:, 1]
    v_w = velocities[:, 2]
    v_h = velocities[:, 3]
    xc = (w_p / (math.sqrt(2) * sigma) * v_xc + xc_p).unsqueeze(-1)
    yc = (h_p / (math.sqrt(2) * sigma) * v_yc + yc_p).unsqueeze(-1)
    w = (w_p * torch.exp(v_w)).unsqueeze(-1)
    h = (h_p * torch.exp(v_h)).unsqueeze(-1)
    x = xc - 0.5 * w
    y = yc - 0.5 * h
    return torch.cat([x, y, w, h], -1)
