import math
import torch
import torch.nn.functional as F
from torch import nn
from .base_transformer import Transformer, LayerNorm


def build_clipvit(cfg):
    if cfg.MODEL.VISUAL_MODEL == "clipvit":
        
        kwargs = {
            "embed_dim": 512,
            "input_resolution": (384,128)
        }
        model = visual_transformer_B32(**kwargs)
#         model = visual_transformer_B16(**kwargs)

        return model

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb
    


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int or tuple, patch_size: int, width: int, layers: int, heads: int, embed_dim: int, checkpoint: bool, dropout: float=0, emb_dropout: float=0):
        super().__init__()
        self.input_resolution = input_resolution
        output_dim = embed_dim
        self.output_dim = output_dim
        self.freeze_conv1 = True
        # self.freeze_conv1 = False
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        
        if isinstance(input_resolution, int):
            self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        if isinstance(input_resolution, tuple):
            self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution[0] // patch_size) * (input_resolution[1] // patch_size) + 1, width))   
            
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, checkpoint=checkpoint, dropout=dropout, emb_dropout=emb_dropout)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.initialize_parameters()
        
        pretrained = True
        
        if pretrained:
            self.load_param()
            
    def load_param(self):
        path = './pretrained/clip/declip_vitb32_convert.pth.tar'
        path = './pretrained/clip/ViT-B-16_visual.pt'
        path = './pretrained/clip/ViT-B-32_visual.pt'
        checkpoint = torch.load(path)
#         self.load_state_dict(checkpoint['state_dict'])
        for k, v in checkpoint.items():
            if k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                print(111)
                v = resize_pos_embed(v, self.positional_embedding, self.input_resolution[0], self.input_resolution[1])
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
                
#         self.load_state_dict(checkpoint)
        print('Loading pretrained model from {}'.format(path))

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_conv1:
            print('-----------------------------------------------------------')
            for layer in [self.conv1]:
                print('set conv1.requires_grad to False')
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
        return self


    def forward(self, x: torch.Tensor, return_dense=False, return_feature=False):
        
        return_feature = True
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        dense_feat = x[:, 1:, :]
        x = self.ln_post(x[:, 0, :])
        feature = x

        if self.proj is not None:
            x = x @ self.proj
        

        ret = [x]
        if return_dense:
            ret.append(dense_feat)
        if return_feature:
            ret.append(feature)
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)
        # if return_dense:
        #     return x, dense_feat

        # return x

def visual_transformer_B32(**kwargs):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64

    default_kwargs = {
        # 'output_dim': 512, from config
        'layers':vision_layers,
        'heads': vision_heads, 
        'input_resolution': 224,
        'patch_size': 32,
        'width': vision_width,
        'checkpoint': False
    }
    default_kwargs.update(**kwargs)
    model = VisualTransformer(**default_kwargs)
    return model

def visual_transformer_B16(**kwargs):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64

    default_kwargs = {
        # 'output_dim': 512, from config
        'layers':vision_layers,
        'heads': vision_heads,
        'input_resolution': 224,
        'patch_size': 16,
        'width': vision_width,
        'checkpoint': False
    }
    default_kwargs.update(**kwargs)
    model = VisualTransformer(**default_kwargs)
    return model
    
