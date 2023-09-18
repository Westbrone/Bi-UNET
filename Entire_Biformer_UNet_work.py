import copy
import logging
import torch.nn as nn
import torch
import pathlib
from .try_Bi_former_Unet import BasicLayer_decoder as BiUnet


logger = logging.getLogger(__name__)


class Bi_Unet(nn.Module):
    def __init__(self, config, num_classes=2, zero_head=False):
        super(Bi_Unet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.BiUnet = BiUnet()

    def forward(self, x):
        evice = torch.device('cuda:0')
        x = x.to(evice)
        logits = self.BiUnet(x)
        return logits



    def load_from(self, config):
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of bi encoder---")

            model_dict = self.BiUnet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "downsample_layers." in k:
                    current_layer_num = 3-int(k[18:19])
                    current_k = "downsample_layers." + str(current_layer_num) + k[19:]
                    full_dict.update({current_k:v})
                if "stages." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "stages." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        #print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.BiUnet.load_state_dict(full_dict, strict=False)
            print("yes!!! pre_model load!!! finish")
        else:
            print("none pretrain")

