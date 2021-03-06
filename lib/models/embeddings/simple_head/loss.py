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
        self.projection_text = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
        self,
        visual_embed,
        textual_embed,
        captions,
        fine = False,
        text_to_image_sim = None,
        image_to_text_sim = None,
        global_cdcr_loss = None,
        
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        loss = {
            "triplt_loss": losses.triplt_loss(
                self.projection,
                visual_embed,
                textual_embed,
                labels,
                epsilon=self.epsilon,
            ),  
#             "instance_loss": losses.instance_loss(
#                 self.projection,
#                 self.projection_text,
#                 visual_embed,
#                 textual_embed,
#                 labels,
#                 epsilon=self.epsilon,
#             ),              
            "global_align_loss": losses.global_align_loss(
                visual_embed,
                textual_embed,
                labels,
                scale_pos=self.scale_pos,
                scale_neg=self.scale_neg,
            ) * 1.5,
        }
        if fine:
            assert text_to_image_sim != None and image_to_text_sim != None, "local needs sim!"
#             loss["global_align_local_loss"] = 0.75 * losses.global_align_loss_from_sim(
#                 image_to_text_sim,
#                 labels,
#                 scale_pos=self.scale_pos,
#                 scale_neg=self.scale_neg,
#             )
            
            loss["global_align_local_loss"] = 1/2 * (losses.global_align_loss_from_sim(
                text_to_image_sim,
                labels,
                scale_pos=self.scale_pos,
                scale_neg=self.scale_neg,
            ) + losses.global_align_loss_from_sim(
                image_to_text_sim,
                labels,
                scale_pos=self.scale_pos,
                scale_neg=self.scale_neg,
            ))
            
            
#             loss["global_align_local_loss"] = losses.global_align_loss_from_sim(
#                 text_to_image_sim.t(),
#                 labels,
#                 scale_pos=self.scale_pos,
#                 scale_neg=self.scale_neg,
#             )
            if global_cdcr_loss != None:
                loss["global_cdcr_loss"] = global_cdcr_loss
            
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
