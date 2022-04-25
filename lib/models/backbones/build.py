from .gru import build_gru
from .text_vit import build_text_vit
from .m_resnet import build_m_resnet
from .resnet import build_resnet
from .transformer import build_vit
from .clip import build_clipvit
from .share_block import build_share_block

def build_visual_model(cfg):
    if cfg.MODEL.VISUAL_MODEL in ["resnet50", "resnet101"]:
        return build_resnet(cfg)
    if cfg.MODEL.VISUAL_MODEL in ["m_resnet50", "m_resnet101"]:
        return build_m_resnet(cfg)
    if cfg.MODEL.VISUAL_MODEL in ["vit"]:
        return build_vit(cfg)
    if cfg.MODEL.VISUAL_MODEL in ["clipvit"]:
        return build_clipvit(cfg)
    raise NotImplementedError


def build_textual_model(cfg,share_block=None):
    if cfg.MODEL.TEXTUAL_MODEL == "bigru":
        return build_gru(cfg, bidirectional=True)
    elif cfg.MODEL.TEXTUAL_MODEL == "vit":
        return build_text_vit(cfg,share_block=share_block)
    raise NotImplementedError
    
def build_share_block_model(cfg):
    return build_share_block(cfg)
