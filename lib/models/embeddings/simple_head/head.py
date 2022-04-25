import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import numpy as np
from .loss import make_loss_evaluator

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(mask, 0.)    
    numer = t.sum(dim = dim)
    mask = ~mask
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def mean(t, dim = -1, eps = 1e-6):
#     t = t.masked_fill(mask, 0.)
    numer = t.sum(dim = dim)
#     denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / t.shape[-1]

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

class SimpleHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        
        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, captions):
        batch_size = visual_feature.size(0)

        visual_embed = visual_feature.view(batch_size, -1)
        textual_embed = textual_feature.view(batch_size, -1)

        visual_embed = self.visual_embed_layer(visual_embed)
        textual_embed = self.textual_embed_layer(textual_embed)

        if self.training:
            losses = self.loss_evaluator(visual_embed, textual_embed, captions)
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        return outputs, None


def build_simple_head(cfg, visual_size, textual_size):
    model = SimpleHead(cfg, visual_size, textual_size)
    return model

def build_simple_vit_head(cfg, visual_size, textual_size):
    model = SimpleVitHead(cfg, visual_size, textual_size)
    return model

def build_simple_fine_head(cfg, visual_size, textual_size):
    model = SimpleFineGrainedHead(cfg, visual_size, textual_size)
    return model


class SimpleVitHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        
        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, captions):
        
#         if self.training:
#             cls_score, visual_feature = visual_feature
        
        batch_size = visual_feature.size(0)        
        visual_embed = visual_feature.view(batch_size, -1)
        textual_embed = textual_feature.view(batch_size, -1)

        visual_embed = self.visual_embed_layer(visual_embed)
        textual_embed = self.textual_embed_layer(textual_embed)

        if self.training:
            losses = self.loss_evaluator(visual_embed, textual_embed, captions)
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        return outputs, None

class SimpleFineGrainedHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        
        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)
        
        self.visual_embed_layer_tokens = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer_tokens = nn.Linear(textual_size, self.embed_size)

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, captions):
        
        visual_feature_cls, visual_feature_tokens = visual_feature
        textual_feature_cls, textual_feature_tokens = textual_feature
        

        visual_embed_cls = self.visual_embed_layer(visual_feature_cls)
        textual_embed_cls = self.textual_embed_layer(textual_feature_cls)
        
        visual_embed_tokens = self.visual_embed_layer_tokens(visual_feature_tokens)
        textual_embed_tokens = self.textual_embed_layer_tokens(textual_feature_tokens)
        
        visual_embed_tokens, textual_embed_tokens = map(l2norm, (visual_embed_tokens, textual_embed_tokens))
        
        
        
        """
        b -  batch size of visual_embed_tokens
        q -  batch size of textual_embed_tokens
        v - length of visual_embed_tokens
        t - length of textual_embed_tokens
        d - dimension of each token
        """
        
        # compute fine-grained similarity
        
        text_length = torch.stack([caption.length for caption in captions], dim=1)
        text_length = text_length.view(-1)
        B,L,D = textual_embed_tokens.shape
        text_mask = torch.zeros(B,B,L)
        for i in range(B):
            text_mask[i,:,text_length[i]:] = 1  
        device = visual_embed_tokens.device
        text_mask = text_mask.to(torch.bool).to(device)
        
        sim_image_to_text = einsum('b v d, q t d -> b q v t', [visual_embed_tokens, textual_embed_tokens])
        image_to_text = reduce(sim_image_to_text, '... v i -> ... v', 'max')
#         image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m = num_batch_texts)
        image_to_text_sim = mean(image_to_text, dim = -1)
        
        # text_imnage
        text_to_image = reduce(sim_image_to_text, '... v i -> ... i', 'max')
#         image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m = num_batch_texts)
        text_to_image_sim = masked_mean(text_to_image, mask=text_mask, dim = -1)
#         text_to_image_sim = mean(text_to_image, dim = -1)

        
#         print(text_to_image_sim)
#         print(image_to_text_sim)
        if self.training:
            losses = self.loss_evaluator(visual_embed_cls, textual_embed_cls, captions, 
                    fine=True, text_to_image_sim=text_to_image_sim, image_to_text_sim=image_to_text_sim)
            return None, losses

        outputs = list()
        outputs.append(visual_embed_cls)
        outputs.append(textual_embed_cls)
        outputs.append(visual_embed_tokens)
        outputs.append(textual_embed_tokens)
        outputs.append(text_mask[:,0,:])
        return outputs, None
    
class SimpleFineGrainedHead2(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        
        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)
        
        self.visual_embed_layer_tokens = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer_tokens = nn.Linear(textual_size, self.embed_size)

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, captions):
        
        visual_feature_cls, visual_feature_tokens = visual_feature
        textual_feature_cls, textual_feature_tokens = textual_feature
        
#         batch_size = visual_feature_cls.size(0)        
#         visual_embed = visual_feature_cls.view(batch_size, -1)
#         textual_embed = textual_feature_cls.view(batch_size, -1)

        visual_embed_cls = self.visual_embed_layer(visual_feature_cls)
        textual_embed_cls = self.textual_embed_layer(textual_feature_cls)
        
        visual_embed_tokens = self.visual_embed_layer_tokens(visual_feature_tokens)
        textual_embed_tokens = self.textual_embed_layer_tokens(textual_feature_tokens)
        
        visual_embed_local, textual_embed_local = map(l2norm, (visual_embed_tokens, textual_embed_tokens))
        
        visual_embed_global = l2norm(visual_embed_cls)
        textual_embed_global = l2norm(textual_embed_cls)
                
        
        """
        b -  batch size of visual_embed_tokens
        q -  batch size of textual_embed_tokens
        v - length of visual_embed_tokens
        t - length of textual_embed_tokens
        d - dimension of each token
        """
#         print(visual_embed_local)
#         print(visual_embed_global)
        visual_embed_sim = einsum('b v d, b d -> b v', [visual_embed_local, visual_embed_global])
        textual_embed_sim = einsum('b t d, b d -> b t', [textual_embed_local, textual_embed_global])
        
        
        _, indices_visual = torch.topk(visual_embed_sim,k=50,dim=-1)
        _, indices_textual = torch.topk(textual_embed_sim,k=50,dim=-1)

        visual_embed_local = visual_embed_local[torch.arange(visual_embed_local.shape[0]).unsqueeze(-1), indices_visual]
        textual_embed_local = textual_embed_local[torch.arange(textual_embed_local.shape[0]).unsqueeze(-1), indices_textual]

        
        # compute fine-grained similarity
        
        sim_image_to_text = einsum('b v d, q t d -> b q v t', [visual_embed_local, textual_embed_local])
        image_to_text = reduce(sim_image_to_text, '... v i -> ... v', 'max')
        image_to_text_sim = mean(image_to_text, dim = -1)
        
        # text_imnage
        text_to_image = reduce(sim_image_to_text, '... v i -> ... i', 'max')
#         image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m = num_batch_texts)
        text_to_image_sim = mean(text_to_image, dim = -1)
        
#         print(text_to_image_sim)
#         print(image_to_text_sim)
        if self.training:
            losses = self.loss_evaluator(visual_embed_cls, textual_embed_cls, captions, 
                    fine=True, text_to_image_sim=text_to_image_sim, image_to_text_sim=image_to_text_sim)
            return None, losses

        outputs = list()
        outputs.append(visual_embed_cls)
        outputs.append(textual_embed_cls)
        outputs.append(visual_embed_local)
        outputs.append(textual_embed_local)
        return outputs, None
    
class SimpleFineGrainedHead3(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        
        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)
        
        self.visual_embed_layer_tokens = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer_tokens = nn.Linear(textual_size, self.embed_size)

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, captions):
        
        visual_feature_cls, visual_feature_tokens = visual_feature
        textual_feature_cls, textual_feature_tokens = textual_feature
        

        visual_embed_cls = self.visual_embed_layer(visual_feature_cls)
        textual_embed_cls = self.textual_embed_layer(textual_feature_cls)
        
        visual_embed_tokens = self.visual_embed_layer_tokens(visual_feature_tokens)
        textual_embed_tokens = self.textual_embed_layer_tokens(textual_feature_tokens)
        
        visual_embed_tokens, textual_embed_tokens = map(l2norm, (visual_embed_tokens, textual_embed_tokens))
        
        
        
        """
        b -  batch size of visual_embed_tokens
        q -  batch size of textual_embed_tokens
        v - length of visual_embed_tokens
        t - length of textual_embed_tokens
        d - dimension of each token
        """
        
        # compute fine-grained similarity
        
        text_length = torch.stack([caption.length for caption in captions], dim=1)
        text_length = text_length.view(-1)
        B,L,D = textual_embed_tokens.shape
        text_mask = torch.ones(B,L)
        for i in range(B):
            text_mask[i,text_length[i]:] = 0  
        device = visual_embed_tokens.device
        text_mask = text_mask.to(device)
                
        retrieve_logits = einsum('b v d, q t d -> b q v t', [visual_embed_tokens, textual_embed_tokens])
        retrieve_logits = torch.einsum('b q v t, q t->b q v t', [retrieve_logits, text_mask])
        text_sum = text_mask.sum(-1)
        visual_sum = retrieve_logits.shape[-2]
        
#         print(text_sum.unsqueeze(0).shape)
        
        t2v_logits, max_idx1 = retrieve_logits.max(dim=-2)  # b q v t -> b q t
        v2t_logits, max_idx2 = retrieve_logits.max(dim=-1)  # b q v t -> b q v
#         print(torch.sum(t2v_logits, dim=2).shape)
        t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(0)) # b q t -> bq
        v2t_logits = torch.sum(v2t_logits, dim=2) / (visual_sum) # b q v -> bq
        text_to_image_sim = t2v_logits
        image_to_text_sim = v2t_logits
        
#         print(text_to_image_sim)
#         print(image_to_text_sim)
        if self.training:
            losses = self.loss_evaluator(visual_embed_cls, textual_embed_cls, captions, 
                    fine=True, text_to_image_sim=text_to_image_sim, image_to_text_sim=image_to_text_sim)
            return None, losses

        outputs = list()
        outputs.append(visual_embed_cls)
        outputs.append(textual_embed_cls)
        outputs.append(visual_embed_tokens)
        outputs.append(textual_embed_tokens)
        outputs.append(text_mask)
        return outputs, None
    
class SimpleFineGrainedHead4(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        
        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)
        
        self.visual_embed_layer_tokens = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer_tokens = nn.Linear(textual_size, self.embed_size)

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, captions):
        
        visual_feature_cls, visual_feature_tokens = visual_feature
        textual_feature_cls, textual_feature_tokens = textual_feature
        

        visual_embed_cls = self.visual_embed_layer(visual_feature_cls)
        textual_embed_cls = self.textual_embed_layer(textual_feature_cls)
        
        visual_embed_tokens = self.visual_embed_layer_tokens(visual_feature_tokens)
        textual_embed_tokens = self.textual_embed_layer_tokens(textual_feature_tokens)
        
        visual_embed_tokens, textual_embed_tokens = map(l2norm, (visual_embed_tokens, textual_embed_tokens))
        
        
        
        """
        q -  batch size of visual_embed_tokens
        b -  batch size of textual_embed_tokens
        v - length of visual_embed_tokens
        t - length of textual_embed_tokens
        d - dimension of each token
        """
        
        # compute fine-grained similarity
        
        text_length = torch.stack([caption.length for caption in captions], dim=1)
        text_length = text_length.view(-1)
        B,L,D = textual_embed_tokens.shape
        text_mask = torch.ones(B,L)
        for i in range(B):
            text_mask[i,text_length[i]:] = 0  
        device = visual_embed_tokens.device
        text_mask = text_mask.to(device)
                
        retrieve_logits = einsum('b t d, q v d -> b q t v', [textual_embed_tokens, visual_embed_tokens])
#         retrieve_logits = torch.einsum('b q t v, b t->b q t v', [retrieve_logits, text_mask])
#         text_sum = text_mask.sum(-1)
        text_sum = retrieve_logits.shape[-1]
        visual_sum = retrieve_logits.shape[-2]
        
#         print(text_sum.unsqueeze(0).shape)
        
        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # b q t v -> b q t
        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # b q v t -> b q v
#         print(torch.sum(t2v_logits, dim=2).shape)
#         t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1)) # b q t -> bq
        t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum) # b q t -> bq
        v2t_logits = torch.sum(v2t_logits, dim=2) / (visual_sum) # b q v -> bq

        
        text_to_image_sim = 1/2 * (t2v_logits + v2t_logits)
        image_to_text_sim = text_to_image_sim
        
        if self.training:
            losses = self.loss_evaluator(visual_embed_cls, textual_embed_cls, captions, 
                    fine=True, text_to_image_sim=text_to_image_sim, image_to_text_sim=image_to_text_sim)
            return None, losses

        outputs = list()
        outputs.append(visual_embed_cls)
        outputs.append(textual_embed_cls)
        outputs.append(visual_embed_tokens)
        outputs.append(textual_embed_tokens)
        outputs.append(text_mask)
        return outputs, None
    
    