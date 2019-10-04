import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision

from lib.resnet_atrous import resnet18
from lib.utils import load_pretrained_weights_to_modified_resnet


class PropagationNetwork(nn.Module):
    def __init__(self):
        super(PropagationNetwork, self).__init__()

        self.POOLING_SIZE = 7  # the ROIs are split into m x m regions
        self.FIXED_BLOCKS = 1
        self.TRUNCATED = False

        # load pretrained weights
        resnet = resnet18()
        resnet_weights = model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
        load_pretrained_weights_to_modified_resnet(resnet, resnet_weights)

        input_channels = 2
        base = [
            #nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.relu,
            resnet.layer2,
            resnet.relu,
            resnet.layer3,
            resnet.relu,
            resnet.layer4,
            resnet.relu]
        self.base = nn.Sequential(*base)

        # # fix some layers
        # def set_bn_fix(m):
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm') != -1:
        #         for p in m.parameters(): p.requires_grad = False
        #
        # self.base.apply(set_bn_fix)

        assert (0 <= self.FIXED_BLOCKS <= 4) # set this value to 0, so we can train all blocks
        if self.FIXED_BLOCKS >= 4: # fix all blocks
            for p in self.base[10].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 3: # fix first 3 blocks
            for p in self.base[8].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 2: # fix first 2 blocks
            for p in self.base[6].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 1: # fix first 1 block
            for p in self.base[4].parameters(): p.requires_grad = False

        self.conv1x1 = nn.Conv2d(512, 4*self.POOLING_SIZE*self.POOLING_SIZE, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.pooling = nn.AvgPool2d(kernel_size=self.POOLING_SIZE, stride=self.POOLING_SIZE)

        self._init_weights()

        print([p.requires_grad for p in self.base.parameters()])
        print(list(self.children()))


    def forward(self, motion_vectors, boxes_prev, num_boxes_mask, motion_vector_scale):
        #print(boxes_prev)
        #print(motion_vector_scale)
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
        #boxes_prev[..., :, 0] = boxes_prev[..., :, 0] - boxes_prev[..., 0, 0]
        boxes_prev = self._frame_idx_to_batch_idx(boxes_prev)
        #print(boxes_prev)

        # compute ratio of input size to size of base output
        spatial_scale = 1/16 * motion_vector_scale
        x = torchvision.ops.ps_roi_pool(x, boxes_prev, output_size=(self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=spatial_scale)
        #print(x)
        #print("after roi_pool", x.shape)
        x = self.pooling(x)
        velocities_pred = x.squeeze()
        #print("after averaging", velocities_pred.shape)

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


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        # init the box regression conv layer
        normal_init(self.conv1x1, 0, 0.001, self.TRUNCATED)

        # init the first conv layer of rcnn_base_mv
        normal_init(self.base[0], 0, 0.01, self.TRUNCATED)


if __name__ == "__main__":

    model = PropagationNetwork()
    #print([p.requires_grad for p in model.base.parameters()])
    #print(list(model.children()))

    # input = torch.zeros(1, 2, 1080, 1920)
    # output = model(input)
    # print(input.shape)
    # print(output.shape)
