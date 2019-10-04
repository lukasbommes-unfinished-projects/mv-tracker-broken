import cv2
#import numpy as np

from lib.visu import draw_boxes
from lib.dataset.velocities import box_from_velocities


class Visualizer:

    def save_inputs(self, motion_vectors, boxes_prev, num_boxes_mask, motion_vector_scale, velocities):
        motion_vectors = motion_vectors.cpu()
        boxes_prev = boxes_prev.cpu()
        num_boxes_mask = num_boxes_mask.cpu()
        motion_vector_scale = motion_vector_scale.cpu()
        velocities = velocities.cpu()

        # prepare motion vectors
        batch_idx = 0
        motion_vectors = motion_vectors[batch_idx, ...]
        motion_vectors = motion_vectors[[2, 1, 0], ...]
        motion_vectors = motion_vectors.permute(1, 2, 0)
        motion_vectors = motion_vectors.numpy()

        num_boxes_mask = num_boxes_mask[batch_idx, ...]

        # show boxes_prev
        boxes_prev = boxes_prev[batch_idx, ...]
        boxes_prev = boxes_prev[num_boxes_mask]
        boxes_prev = boxes_prev[..., 1:5]
        boxes_prev_scaled = boxes_prev * motion_vector_scale[0, 0]
        motion_vectors = draw_boxes(motion_vectors, boxes_prev_scaled.numpy(), None, color=(200, 200, 200))

        # compute boxes based on velocities and boxes_prev
        velocities = velocities[batch_idx, ...]
        velocities = velocities[num_boxes_mask]
        boxes = box_from_velocities(boxes_prev, velocities)

        #print(velocities)
        #print(boxes_prev)
        #print(boxes)

        # show boxes
        boxes_scaled = boxes * motion_vector_scale[0, 0]
        motion_vectors = draw_boxes(motion_vectors, boxes_scaled.numpy(), None, color=(255, 0, 0))

        # store frame to write ouputs into it
        self.motion_vectors = motion_vectors


    def save_outputs(self, velocities_pred):
        print(velocities_pred.shape)
        print(velocities_pred)
        # TODO: convert into format [B, N, 4]
        # compute predicted bounding boxes based on the predicted velocities
        # show predicted boxes


    def show(self):
        cv2.imshow("motion_vectors", self.motion_vectors)
        key = cv2.waitKey(1)
