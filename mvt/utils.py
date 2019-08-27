import cv2
import numpy as np


def draw_motion_vectors(frame, motion_vectors):
    if np.shape(motion_vectors)[0] > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 5], mv[0, 6])
            end_pt = (mv[0, 3], mv[0, 4])
            if mv[0, 0] < 0:
                cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.3)
            else:
                cv2.arrowedLine(frame, start_pt, end_pt, (0, 255, 0), 1, cv2.LINE_AA, 0, 0.3)
    return frame


def draw_boxes(frame, bounding_boxes, color=(0, 255, 0)):
    """Draw bounding boxes and detected positions.
    Function draws bounding boxes around tracked objects for each frame in the frames list.
    Furthermore, the object positions are drawn as the middle of the bottom edge of each
    bounding box. Boxes are drawn orange as long as there state is not yet valid. As soon
    as a box is valid, it is drawn in green. Each box has an accoridng boy ID shown on top
    of the box. The orange number at the bottom of each boy indicates the state of the boxes
    associated match counter, which controls, whether the box is valid is invalid (and thus
    being deleted). The parameters draw_box_id, draw_invalid_boxes, draw_match_counter control
    what elements are drawn.
    """
    for box in bounding_boxes:
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[0] + box[2])
        ymax = int(box[1] + box[3])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_4)
        #cv2.putText(frame, '{}'.format(box_id), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def draw_box_ids(frame, bounding_boxes, box_ids, color=(0, 0, 255)):
    for box, box_id in zip(bounding_boxes, box_ids):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[0] + box[2])
        ymax = int(box[1] + box[3])
        cv2.putText(frame, '{}'.format(str(box_id)[:6]), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return frame


def draw_shifts(frame, shifts, bounding_boxes, color=(255, 255, 255)):
    if shifts is not None:
        for box, shift in zip(bounding_boxes, shifts):
            start_pt = (int(box[0] + box[2]/2), int(box[1] + box[3]/2))
            end_pt = (int(start_pt[0] + shift[0]), int(start_pt[1] + shift[1]))
            cv2.arrowedLine(frame, start_pt, end_pt, color, 1, cv2.LINE_AA, 0, 0.3)

    return frame
