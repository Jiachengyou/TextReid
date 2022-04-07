import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses
import random

class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.epsilon = cfg.MODEL.EMBEDDING.EPSILON
        self.scale_pos = 10.0
        self.scale_neg = 40.0

        self.projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
        self,
        visual_embed,
        textual_embed,
        captions,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        if random.uniform(0, 1) >= 0.5:
            visual_embed_t = visual_embed
            textual_embed_t = textual_embed
        else:
            visual_embed_t = visual_embed
            textual_embed_t = textual_embed
        loss = {
            "instance_loss": losses.instance_loss(
                self.projection,
                visual_embed_t,
                textual_embed_t,
                labels,
                epsilon=self.epsilon,
            ),           
            
            "global_align_loss": losses.global_align_loss(
                visual_embed_t,
                textual_embed_t,
                labels,
                scale_pos=self.scale_pos,
                scale_neg=self.scale_neg,
            ),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
