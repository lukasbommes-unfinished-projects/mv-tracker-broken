import cv2
import numpy as np


# for MVS as numpy arrays
# def draw_motion_vectors(frame, motion_vectors):
#     if np.shape(motion_vectors)[0] > 0:
#         num_mvs = np.shape(motion_vectors)[0]
#         for mv in np.split(motion_vectors, num_mvs):
#             start_pt = (mv[0, 5], mv[0, 6])
#             end_pt = (mv[0, 3], mv[0, 4])
#             if mv[0, 0] < 0:
#                 cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.3)
#             else:
#                 cv2.arrowedLine(frame, start_pt, end_pt, (0, 255, 0), 1, cv2.LINE_AA, 0, 0.3)
#     return frame


def draw_motion_vectors(frame, motion_vectors):
    if motion_vectors.shape[0] > 0:
        for x in range(motion_vectors.shape[2]):
            for y in range(motion_vectors.shape[1]):
                motion_x = motion_vectors[0, y, x]
                motion_y = motion_vectors[1, y, x]
                end_pt = (x * 16 + 8, y * 16 + 8)  # the x,y coords correspond to the vector destination
                start_pt = (end_pt[0] + motion_x, end_pt[1] + motion_y)  # vector source
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
