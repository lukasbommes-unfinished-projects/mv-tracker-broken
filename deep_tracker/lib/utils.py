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


# TESTING
if __name__ == "__main__":
    import numpy as np

    gt_boxes_prev = np.array([[1338.,  418.,  167.,  379.],
              [ 586.,  447.,   85.,  263.],
              [1416.,  431.,  184.,  336.],
              [1056.,  484.,   36.,  110.],
              [1091.,  484.,   31.,  115.],
              [1255.,  447.,   33.,  100.],
              [1016.,  430.,   40.,  116.],
              [1101.,  441.,   38.,  108.],
              [ 935.,  436.,   42.,  114.],
              [ 442.,  446.,  105.,  283.],
              [ 636.,  458.,   61.,  187.],
              [1364.,  434.,   51.,  124.],
              [1478.,  434.,   63.,  124.],
              [ 473.,  460.,   89.,  249.],
              [ 548.,  465.,   35.,   93.],
              [ 418.,  459.,   40.,   84.],
              [ 582.,  456.,   35.,  133.],
              [ 972.,  456.,   32.,   77.],
              [ 578.,  432.,   20.,   43.],
              [ 596.,  429.,   18.,   42.],
              [ 663.,  451.,   34.,   86.]])

    gt_boxes = np.array([[1342.,  417.,  168.,  380.],
              [ 586.,  446.,   85.,  264.],
              [1422.,  431.,  183.,  337.],
              [1055.,  483.,   36.,  110.],
              [1090.,  484.,   32.,  114.],
              [1255.,  447.,   33.,  100.],
              [1015.,  430.,   40.,  116.],
              [1100.,  440.,   38.,  108.],
              [ 934.,  435.,   42.,  114.],
              [ 442.,  446.,  107.,  282.],
              [ 636.,  458.,   61.,  187.],
              [1365.,  434.,   52.,  124.],
              [1480.,  433.,   62.,  125.],
              [ 473.,  460.,   89.,  249.],
              [ 547.,  464.,   35.,   93.],
              [ 418.,  459.,   40.,   84.],
              [ 582.,  455.,   34.,  134.],
              [ 972.,  456.,   32.,   77.],
              [ 578.,  431.,   20.,   43.],
              [ 595.,  428.,   18.,   42.],
              [1035.,  452.,   25.,   67.],
              [ 664.,  451.,   34.,   85.]])

    gt_ids = np.array([ 2.,  3.,  8.,  9., 10., 14., 15., 17., 18., 19., 20., 21., 22., 23.,
              26., 31., 36., 39., 68., 69., 70., 72.])

    gt_ids_prev = np.array([ 2.,  3.,  8.,  9., 10., 14., 15., 17., 18., 19., 20., 21., 22., 23.,
              26., 31., 36., 39., 68., 69., 72.])

    _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev, assume_unique=True, return_indices=True)
    print(idx_1, idx_0)
    boxes = torch.from_numpy(gt_boxes[idx_1])
    boxes_prev = torch.from_numpy(gt_boxes_prev[idx_0])

    print(boxes.shape)
    print(boxes_prev.shape)

    print("boxes_prev:", boxes_prev)

    velocities = velocities_from_boxes(boxes_prev, boxes)
    print(velocities)
    print(velocities.shape)


    box = box_from_velocities(boxes_prev, velocities)
    print(box)
    print(box.shape)
    print(boxes)
