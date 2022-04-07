from yacs.config import CfgNode as CN

_C = CN()
_C.ROOT = "./"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST = ()
_C.DATASETS.USE_ONEHOT = True


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.IMS_PER_ID = 4
_C.DATALOADER.EN_SAMPLER = True


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.HEIGHT = 224
_C.INPUT.WIDTH = 224
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.PADDING = 10
_C.INPUT.USE_AUG = False


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.VISUAL_MODEL = "resnet50"
_C.MODEL.TEXTUAL_MODEL = "bilstm"
_C.MODEL.NUM_CLASSES = 11003
_C.MODEL.FREEZE = False
_C.MODEL.WEIGHT = "imagenet"


# -----------------------------------------------------------------------------
# VIT
# -----------------------------------------------------------------------------
_C.MODEL.VIT = CN()
_C.MODEL.VIT.NAME = 'transformer'
_C.MODEL.VIT.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.VIT.PRETRAIN_PATH = './pretrained/clip/jx_vit_base_p16_224-80ecf9dd.pth'

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.VIT.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.VIT.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.VIT.IF_WITH_CENTER = 'no'

_C.MODEL.VIT.ID_LOSS_TYPE = 'softmax'
_C.MODEL.VIT.ID_LOSS_WEIGHT = 1.0
_C.MODEL.VIT.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.VIT.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.VIT.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.VIT.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.VIT.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.VIT.COS_LAYER = False

# Transformer setting
_C.MODEL.VIT.DROP_PATH = 0.1
_C.MODEL.VIT.DROP_OUT = 0.0
_C.MODEL.VIT.ATT_DROP_RATE = 0.0
_C.MODEL.VIT.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
_C.MODEL.VIT.STRIDE_SIZE = [16, 16]

#SIE
_C.MODEL.VIT.SIE_COE = 3.0
_C.MODEL.VIT.SIE_CAMERA = False
_C.MODEL.VIT.SIE_VIEW = False

# add
_C.MODEL.VIT.NECK_FEAT = 'before'


# -----------------------------------------------------------------------------
# MoCo
# -----------------------------------------------------------------------------
_C.MODEL.MOCO = CN()
_C.MODEL.MOCO.K = 1024
_C.MODEL.MOCO.M = 0.999
_C.MODEL.MOCO.FC = True


# -----------------------------------------------------------------------------
# GRU
# -----------------------------------------------------------------------------
_C.MODEL.GRU = CN()
_C.MODEL.GRU.ONEHOT = "yes"
_C.MODEL.GRU.EMBEDDING_SIZE = 512
_C.MODEL.GRU.NUM_UNITS = 512
_C.MODEL.GRU.VOCABULARY_SIZE = 12000
_C.MODEL.GRU.DROPOUT_KEEP_PROB = 0.7
_C.MODEL.GRU.MAX_LENGTH = 100
_C.MODEL.GRU.NUM_LAYER = 1


# -----------------------------------------------------------------------------
# Resnet
# -----------------------------------------------------------------------------
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.RES5_STRIDE = 2
_C.MODEL.RESNET.RES5_DILATION = 1
_C.MODEL.RESNET.PRETRAINED = None


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------
_C.MODEL.EMBEDDING = CN()
_C.MODEL.EMBEDDING.EMBED_HEAD = "simple"
_C.MODEL.EMBEDDING.FEATURE_SIZE = 512
_C.MODEL.EMBEDDING.DROPOUT_PROB = 0.3
_C.MODEL.EMBEDDING.EPSILON = 0.0


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.EVALUATE_PERIOD = 1

_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.BASE_LR = 0.0002
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.WEIGHT_DECAY = 0.00004
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.ADAM_ALPHA = 0.9
_C.SOLVER.ADAM_BETA = 0.999
_C.SOLVER.SGD_MOMENTUM = 0.9

_C.SOLVER.LRSCHEDULER = "step"

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500,)

_C.SOLVER.POWER = 0.9
_C.SOLVER.TARGET_LR = 0.0001


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 16


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #
# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"
# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False
