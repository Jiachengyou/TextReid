import torch
import torch.nn as nn
from .vit_pytorch import Block,trunc_normal_
from lib.utils.directory import load_vocab_dict
from functools import partial
import copy

class TextVit(nn.Module):
    def __init__(
        self,
        cfg,
        vocab_size,
        use_onehot,
        root,
        share_block = None
    ):
        super().__init__()
        
        embed_dim = 768
        self.use_onehot = use_onehot
        
        
        # word embedding
        if use_onehot == "yes":
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        else:
            if vocab_size == embed_dim:
                self.embed = None
            else:
                self.embed = nn.Linear(vocab_size, embed_dim)

            vocab_dict = load_vocab_dict(root, use_onehot)
            assert vocab_size == vocab_dict.shape[1]
            self.vocab_dict = torch.tensor(vocab_dict).cuda().float()
        
        # transformer
        # base version
        depth = 1
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
        
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, norm_first=False)
#             for i in range(depth)]) 
        
#         self.blocks = share_block
#         self.blocks = copy.deepcopy(share_block)
        
        self.out_channels = 768
        self.norm = norm_layer(embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,  dim_feedforward=int(embed_dim * mlp_ratio), norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def forward(self, captions):
        text = torch.stack([caption.text for caption in captions], dim=1)
        text_length = torch.stack([caption.length for caption in captions], dim=1)

        text_length = text_length.view(-1)
        text = text.view(-1, text.size(-1))  # b x l       
        

        if not self.use_onehot == "yes":
            bs, length = text.shape[0], text.shape[-1]
            text = text.view(-1)  # bl
            text = self.vocab_dict[text].reshape(bs, length, -1)  # b x l x vocab_size
        if self.embed is not None:
            text = self.embed(text)
        
        # cls encoding
        B,L,D = text.shape        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        text = torch.cat((cls_tokens, text), dim=1)

        text = text + self.pos_embed

        key_padding_mask = torch.zeros(B,L+1)
        for i in range(B):
            key_padding_mask[i,text_length[i]+1:] = 1
            
        key_padding_mask = key_padding_mask.to(torch.bool).to(text.device)

        
#         text = self.encoder(text.transpose(0,1),src_key_padding_mask=key_padding_mask).transpose(0,1)
#         return x[:, 0]
        return [text,key_padding_mask]
        # self
        x = text
        for blk in self.blocks[-6:]:
            x = blk(x, mask=key_padding_mask, norm_first=False)
#         x = self.norm(x)
#         print(x.shape)
#         print(x[:, 0].shape)
        return x[:, 0]
        out, _ = torch.max(x, dim=1)
#         print(out.shape)
        return out


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def build_text_vit(cfg,share_block = None):  
    use_onehot = cfg.MODEL.GRU.ONEHOT
    vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
    root = cfg.ROOT

    model = TextVit(cfg,
        vocab_size,
        use_onehot,
        root,share_block = share_block)

    if cfg.MODEL.FREEZE:
        for m in [model.embed, model.gru]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
