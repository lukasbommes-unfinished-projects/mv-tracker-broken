class Config(object):
    SCALING_FACTOR = 1.0
    DETECTOR_PATH = "models/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb"  # detector frozen inferenze graph (*.pb)
    DETECTOR_BOX_SIZE_THRES = None #(0.25*1920, 0.6*1080) # discard detection boxes larger than this threshold
    DETECTOR_INTERVAL = 10
    TRACKER_IOU_THRES = 0.3
