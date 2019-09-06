import cv2
import numpy as np


def motion_vectors_to_image(motion_vectors):
    mvs = motion_vectors.numpy()
    image = np.zeros((mvs.shape[1], mvs.shape[2], 3))
    # scale the components to range [0, 1]
    mvs_min = np.min(mvs, axis=(1, 2))
    mvs_max = np.max(mvs, axis=(1, 2))
    if (mvs_max[0] - mvs_min[0]) != 0 and (mvs_max[1] - mvs_min[1]) != 0:
        mvs_x = (mvs[0, :, :] - mvs_min[0]) / (mvs_max[0] - mvs_min[0])
        mvs_y = (mvs[1, :, :] - mvs_min[1]) / (mvs_max[1] - mvs_min[1])
        image[:, :, 2] = mvs_x
        image[:, :, 1] = mvs_y
    return image


def draw_boxes_on_motion_vector_image(mvs_image, bounding_boxes, color=(255, 255, 255)):
    for box in bounding_boxes:
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[0] + box[2])
        ymax = int(box[1] + box[3])
        cv2.rectangle(mvs_image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_4)
    return mvs_image


def draw_motion_vectors(frame, motion_vectors):
    if motion_vectors.shape[0] > 0:
        for x in range(motion_vectors.shape[2]):
            for y in range(motion_vectors.shape[1]):
                motion_x = motion_vectors[0, y, x]
                motion_y = motion_vectors[1, y, x]
                end_pt = (x * 16 + 8, y * 16 + 8)  # the x,y coords correspond to the vector destination
                start_pt = (end_pt[0] - motion_x, end_pt[1] - motion_y)  # vector source
                cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.3)
    return frame


def draw_boxes(frame, bounding_boxes, box_ids=None, color=(0, 255, 0)):
    if box_ids is not None:
        for box, box_id in zip(bounding_boxes, box_ids):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[0] + box[2])
            ymax = int(box[1] + box[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_4)
            cv2.putText(frame, '{}'.format(str(box_id)[:6]), (xmin, ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    else:
        for box in bounding_boxes:
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[0] + box[2])
            ymax = int(box[1] + box[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_4)
    return frame


def draw_velocities(frame, bounding_boxes, velocities):
    for box, velocity in zip(bounding_boxes, velocities):
        box = box*16.0
        start_pt = (int(box[0] + 0.5 * box[2]), int(box[1] + 0.5 * box[3]))
        end_pt = (int(start_pt[0] + 100*velocity[0]), int(start_pt[1] + 100*velocity[1]))
        cv2.arrowedLine(frame, start_pt, end_pt, (255, 255, 255), 1, cv2.LINE_AA, 0, 0.3)
    return frame
