from .simple_head.head import build_simple_head, build_simple_vit_head, build_simple_fine_head


def build_embed(cfg, visual_out_channels, textual_out_channels):

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "simple" :
        return build_simple_head(cfg, visual_out_channels, textual_out_channels)
    elif cfg.MODEL.EMBEDDING.EMBED_HEAD == "vit":
        return build_simple_vit_head(cfg, visual_out_channels, textual_out_channels)
    elif cfg.MODEL.EMBEDDING.EMBED_HEAD == "fine":
        return build_simple_fine_head(cfg, visual_out_channels, textual_out_channels)
    else:
        raise NotImplementedError
