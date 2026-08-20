from .vit import vit
from .sapiens2 import sapiens2


def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'sapiens2':
        return sapiens2(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
