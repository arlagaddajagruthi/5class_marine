import os
from mmseg.models import build_segmentor
from mmcv.utils import Config
from mmcv.utils import get_logger
from mmcv.cnn.utils import revert_sync_batchnorm
from torch import nn

logger = get_logger('mmdet')
logger.setLevel('WARNING')

class GenericModel(nn.Module):
    def __init__(self, in_chans, num_classes, conf_file):
        super(GenericModel, self).__init__()
        cfg = Config.fromfile(conf_file)
        cfg.model.backbone.in_chans = in_chans
        cfg.model.decode_head.num_classes = num_classes
        model = build_segmentor(cfg.model)
        model.init_weights()
        model = revert_sync_batchnorm(model)
        
        self.backbone = model.backbone
        self.decode_head = model.decode_head 

    def forward(self, x):
        return self.decode_head(self.backbone(x))