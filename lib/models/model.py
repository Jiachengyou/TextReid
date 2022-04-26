from torch import nn
import copy
from functools import partial
from .backbones import build_textual_model, build_visual_model, build_share_block_model
from .embeddings import build_embed
from .embeddings.moco_head.head import build_moco_head


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.visual_model = build_visual_model(cfg)
        self.textual_model = build_textual_model(cfg)
        
        share = True
        if share:
            self.share_block_model = build_share_block_model(cfg)

        if cfg.MODEL.EMBEDDING.EMBED_HEAD == "moco":
            self.embed_model = build_moco_head(
                cfg, self.visual_model, self.textual_model
            )
            self.embed_type = "moco"
        elif cfg.MODEL.EMBEDDING.EMBED_HEAD == "vit":
            self.embed_model = build_embed(
                cfg, 768, self.textual_model.out_channels
            )
            self.embed_type = "normal"
        elif cfg.MODEL.EMBEDDING.EMBED_HEAD == "fine":
            self.embed_model = build_embed(
                cfg, 768, self.textual_model.out_channels
            )
            self.embed_type = "normal"
        else:
            self.embed_model = build_embed(
                cfg, self.visual_model.out_channels, self.textual_model.out_channels
            )
            self.embed_type = "normal"
            
        
            
        #share
#         self.text_share_block = copy.deepcopy(share_block)
#         self.visual_share_block = copy.deepcopy(share_block)
        
#         self.text_share_block = share_block
# #         self.visual_share_block = share_block
        
#         self.bottleneck = copy.deepcopy(bottleneck)
        
#         norm = share_block[0].norm1
#         self.norm_text = copy.deepcopy(norm)
#         self.norm_visual = copy.deepcopy(norm)
        
    def forward_share(self, visual, text, mask=None):
        for blk in self.text_share_block:
            visual = blk(visual)
        visual = self.norm_visual(visual)
        visual = self.bottleneck(visual[:,0])
        for blk in self.text_share_block[-6:]:
            text = blk(text)
#             print(text.shape)
        text = self.norm_text(text) 
        
        return visual,text[:, 0]
    def forward(self, images, captions):
        if self.embed_type == "moco":
            return self.embed_model(images, captions)
        
        text = self.textual_model(captions)
        visual = self.visual_model(images)

        
#         visual_feat, textual_feat = self.share_block_model(visual, text)
        visual_feat, textual_feat = visual[1], text
    
        outputs_embed, losses_embed = self.embed_model(
            visual_feat, textual_feat, captions
        )

        if self.training:
            losses = {}
            losses.update(losses_embed)
            return losses

        return outputs_embed


def build_model(cfg):
    return Model(cfg)
