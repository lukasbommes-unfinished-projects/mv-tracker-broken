import torch
import torchvision
from torchvision import datasets, models, transforms, ops

m = 2
featuremap = torch.Tensor(
[[[[0.88,0.44,0.14,0.16,0.37,0.77,0.96,0.27],
   [0.19,0.45,0.57,0.16,0.63,0.29,0.71,0.70],
   [0.66,0.26,0.82,0.64,0.54,0.73,0.59,0.26],
   [0.85,0.34,0.76,0.84,0.29,0.75,0.62,0.25],
   [0.32,0.74,0.21,0.39,0.34,0.03,0.33,0.48],
   [0.20,0.14,0.16,0.13,0.73,0.65,0.96,0.32],
   [0.19,0.69,0.09,0.86,0.88,0.07,0.01,0.48],
   [0.83,0.24,0.97,0.04,0.24,0.35,0.50,0.91]]]])#.permute(0, 1, 3, 2)

featuremap = torch.tensor(
 [[[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]]]])

#featuremap.random_(-1, 1)
# boxes = torch.tensor([[[0., 1342.,  417.,  168.,  380.],
#          [0., 586.,  446.,   85.,  264.],
#          [0., 1055.,  483.,   36.,  110.]],
#         [[1., 1342.,  417.,  168.,  380.],
#          [1.,  586.,  446.,   85.,  264.],
#          [1., 1055.,  483.,   36.,  110.]]], dtype=torch.float64)

#featuremap = torch.cat([featuremap, featuremap], axis=0)

# boxes = torch.tensor([[0, 0, 3,  6,  7],
#                       [0, 0, 3,  6,  7],
#                       [1, 0, 3,  6,  7],
#                       [1, 0, 3,  6,  7]])

# x1, y1, x2, y2
boxes = torch.Tensor([[0, 0, 0, 3.3, 1]])  # both top-left and bottom-right point are included during pooling operation

print(featuremap.shape)

featuremap = featuremap.float()
boxes = boxes.float()

# boxes = [torch.tensor([[0, 3,  6,  7],
#                       [0, 3,  6,  7]]).float(),
#          torch.tensor([[0, 3,  6,  7],
#                       [0, 3,  6,  7]]).float()]

# def _change_box_format(boxes):
#         """Change format of boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
#         boxes[..., 0] = boxes[..., 0]
#         boxes[..., 1] = boxes[..., 1]
#         boxes[..., 2] = boxes[..., 2]
#         boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
#         boxes[..., 4] = boxes[..., 2] + boxes[..., 4]
#         return boxes

#boxes = _change_box_format(boxes)
#print(boxes)

#print(featuremap.shape)
#print(boxes.shape)

roi_out = ops.roi_pool(featuremap, boxes, output_size=(m, m), spatial_scale=1.0)

print(roi_out.shape)
print(roi_out)
