import torch
import torch.nn as nn
from .vit_pytorch import trunc_normal_, DropPath
from lib.utils.directory import load_vocab_dict
from functools import partial
import copy

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
#         print(attn.shape)
        if mask is not None:
            B,L = mask.shape
            mask = mask.view(B, 1, 1, L).expand(-1, self.num_heads, -1, -1)
#             print(mask.shape)
            attn = attn.masked_fill(mask, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # text        
#         self.attn_text = copy.deepcopy(self.attn)
#         self.mlp_text = copy.deepcopy(self.mlp)
        self.norm1_text = copy.deepcopy(self.norm1)
        self.norm2_text = copy.deepcopy(self.norm2)
#         self.drop_path_text = DropPath(0.1)
    
    
    def forward_visual(self,x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))      
        return x
                               
#     def forward_text(self, text, mask):
#         text = self.norm1_text(text + self.drop_path_text(self.attn(text, mask=mask)))
#         text = self.norm2_text(text + self.drop_path_text(self.mlp(text)))
                               
#         return text
    
    def forward_text(self, text, mask):
        text = text + self.drop_path(self.attn(self.norm1(text), mask))
        text = text + self.drop_path(self.mlp(self.norm2(text))) 
                               
        return text

    def forward(self, x, text=False, mask=None):        
        if text:            
            assert mask != None, "mask can't be None!"
            x = self.forward_text(x, mask=mask)
        else:
            x = self.forward_visual(x, mask=mask)
        return x

class ShareBlock(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        
        
        self.embed_head = cfg.MODEL.EMBEDDING.EMBED_HEAD
        # transformer
        depth = 12
        embed_dim = 768
        num_heads = 12
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = cfg.MODEL.VIT.DROP_OUT
        attn_drop_rate = cfg.MODEL.VIT.ATT_DROP_RATE
        drop_path_rate = cfg.MODEL.VIT.DROP_PATH
        max_length = cfg.MODEL.GRU.MAX_LENGTH
        
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        dpr = [drop_path_rate for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)]) 
        
        self.blocks_text = copy.deepcopy(self.blocks)
        
        # image after block
        self.norm = norm_layer(embed_dim)
        self.norm_text = copy.deepcopy(self.norm)
        self.in_planes = 768
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        
        self.apply(self._init_weights)
        
        self.load_param(cfg.MODEL.VIT.PRETRAIN_PATH)

    def forward(self, visual, text):
        # image
        for blk in self.blocks:
            visual = blk(visual)
        visual = self.norm(visual)   
        
        
        # text
        text, mask = text    
        for blk in self.blocks[-12:]:
            text = blk(text, text=True, mask=mask)  
        text = self.norm_text(text)
        
        # fine grained
        if self.embed_head == 'fine':
            return [visual[:,0], visual[:,1:]], ([text[:,0], text[:,1:]])
        
        visual = self.bottleneck(visual[:,0])
        text = text[:,0]
        return visual, text


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        for k, v in param_dict.items():
            if 'blocks' in k:
                try:
                    self.state_dict()[k].copy_(v)            
                except:
                    print('===========================ERROR=========================')
                    print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
            else:
                continue
        print('===========================Load Blocks Param_dict=========================')


def build_share_block(cfg):  
    

    model = ShareBlock(cfg)

#     if cfg.MODEL.FREEZE:
#         for m in [model.embed, model.gru]:
#             m.eval()
#             for param in m.parameters():
#                 param.requires_grad = False

    return model
