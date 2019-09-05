import time
import copy
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from lib.datasets import MotionVectorDataset


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PropagationNetwork(nn.Module):
    def __init__(self):
        super(PropagationNetwork, self).__init__()

        self.POOLING_SIZE = 7  # the ROIs are split into m x m regions

        layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.base = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            layer1,
            nn.ReLU(inplace=True),
            layer2,
            nn.ReLU(inplace=True),
            layer3,
            nn.ReLU(inplace=True),
            layer4,
            nn.ReLU(inplace=True),
        )

        self.conv1x1 = nn.Conv2d(128, 4*self.POOLING_SIZE*self.POOLING_SIZE, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.pooling = nn.AvgPool2d(kernel_size=self.POOLING_SIZE, stride=self.POOLING_SIZE)

        print([p.requires_grad for p in self.parameters()])
        print([p.shape for p in self.parameters()])
        print(list(self.children()))


    def forward(self, motion_vectors, boxes_prev, num_boxes_mask):
        #print(boxes_prev)
        #print("boxes_prev:", boxes_prev.shape)
        #print("motion_vectors:", motion_vectors.shape)
        x = self.base(motion_vectors)
        #print("after ResNet18:", x.shape)
        x = self.conv1x1(x)
        x = F.relu(x)
        #print("after conv1", x.shape)

        boxes_prev = self._change_box_format(boxes_prev)
        boxes_prev = boxes_prev[num_boxes_mask]
        boxes_prev = boxes_prev.view(-1, 5)
        # offset frame_idx so that it corresponds to batch index
        boxes_prev = self._frame_idx_to_batch_idx(boxes_prev)
        #print(boxes_prev)

        # compute ratio of input size to size of base output
        x = torchvision.ops.ps_roi_pool(x, boxes_prev, output_size=(self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=1/2)
        #print("after roi_pool", x.shape)
        x = self.pooling(x)
        velocities_pred = x.squeeze()
        #print("after averaging", velocities_pred.shape)
        #print(velocities_pred.shape)

        return velocities_pred


    def _frame_idx_to_batch_idx(self, boxes):
        """Converts unique frame_idx in first column of boxes into batch index."""
        frame_idxs = torch.unique(boxes[:, 0])
        for batch_idx, frame_idx in enumerate(frame_idxs):
            idx = torch.where(boxes == frame_idx)[0]
            boxes[idx, 0] = batch_idx
        return boxes


    def _change_box_format(self, boxes):
        """Change format of boxes from [idx, x, y, w, h] to [idx, x1, y1, x2, y2]."""
        boxes[..., 0] = boxes[..., 0]
        boxes[..., 1] = boxes[..., 1]
        boxes[..., 2] = boxes[..., 2]
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
        boxes[..., 4] = boxes[..., 2] + boxes[..., 4]
        return boxes



datasets = {x: MotionVectorDataset(root_dir='data', window_length=1, codec="mpeg4", visu=False, mode=x) for x in ["train", "val", "test"]}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=False, num_workers=8) for x in ["train", "val", "test"]}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PropagationNetwork()
model = model.to(device)

#criterion = nn.SmoothL1Loss(reduction='mean')
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)


for step, (motion_vectors, boxes_prev, velocities, num_boxes_mask) in enumerate(dataloaders["train"]):
    motion_vectors = motion_vectors.to(device)
    boxes_prev = boxes_prev.to(device)
    velocities = velocities.to(device)
    num_boxes_mask = num_boxes_mask.to(device)

    mvs_min = torch.min(motion_vectors)
    mvs_max = torch.max(motion_vectors)
    motion_vectors = (motion_vectors - mvs_min) / (mvs_max - mvs_min)

    velocities = velocities[num_boxes_mask]
    velocities = velocities.view(-1, 4)

    #vel_min = torch.min(velocities)
    #vel_max = torch.max(velocities)
    #velocities = (velocities - vel_min) / (vel_max - vel_min)

    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        velocities_pred = model(motion_vectors, boxes_prev, num_boxes_mask)

        loss = criterion(velocities_pred, velocities)

        loss.backward()

        print("Printing grads: ")
        c = 0
        for p in model.parameters():
            if p.grad is not None:
                c += 1
                print(p.grad.data.sum())
        print("Done printing grads {}".format(c))

        a = list(model.parameters())[0].clone()

        optimizer.step()

        b = list(model.parameters())[0].clone()
        print(torch.equal(a.data, b.data))
