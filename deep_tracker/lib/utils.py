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
