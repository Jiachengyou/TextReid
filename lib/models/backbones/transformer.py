import torch
import torch.nn as nn
# from .backbones.resnet import ResNet, Bottleneck
import copy
from .vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
# from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

# def build_transformer(
#     input_resolution,
#     pretrained_path=None,
# ):
#     model = VisonTransformer(
#         img_size = input_resolution,
#         patch_size = 16,
#         stride_size = 16, 
#         in_chans = 3,
#         embed_dim = 768,
#         depth = 12,
#         num_heads = 8,
#         mlp_ratio=4.,
#         qkv_bias=True, 
#         qk_scale=None, 
#         drop_rate=0.0, 
#         attn_drop_rate=0.0,
#         drop_path_rate=0.1,  
#         norm_layer=partial(nn.LayerNorm, eps=1e-6)
#     )
#     if pretrained_path:
#         model.load_param(pretrained_path)
#         print('Loading pretrained ImageNet model......from {}'.format(pretrained_path))

#     return model

def build_vit(cfg):
    if cfg.MODEL.VISUAL_MODEL == "vit":
        camera_num, view_num =0, 0
        model = build_transformer(cfg, camera_num, view_num, __factory_T_type)
        print('===========building transformer===========')
        return model
    
class build_transformer(nn.Module):
    def __init__(self, cfg, camera_num, view_num, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.VIT.LAST_STRIDE
        model_path = cfg.MODEL.VIT.PRETRAIN_PATH
        model_name = cfg.MODEL.VIT.NAME
        pretrain_choice = cfg.MODEL.VIT.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.VIT.COS_LAYER
        self.neck = cfg.MODEL.VIT.NECK
        self.neck_feat = cfg.MODEL.VIT.NECK_FEAT
        self.in_planes = 768
        
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.VIT.TRANSFORMER_TYPE))

        if cfg.MODEL.VIT.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.VIT.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        
        image_size = (cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)
        self.base = factory[cfg.MODEL.VIT.TRANSFORMER_TYPE](img_size=image_size, sie_xishu=cfg.MODEL.VIT.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.VIT.STRIDE_SIZE, drop_path_rate=cfg.MODEL.VIT.DROP_PATH,
                                                        drop_rate= cfg.MODEL.VIT.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.VIT.ATT_DROP_RATE)
        
        if cfg.MODEL.VIT.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = cfg.MODEL.NUM_CLASSES 
        self.ID_LOSS_TYPE = cfg.MODEL.VIT.ID_LOSS_TYPE
        # to be updated
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
