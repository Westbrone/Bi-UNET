import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
# -----------------------------------------------------------------------------
# 数据读取设计
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 128
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'imagenet'
_C.DATA.IMG_SIZE = 224
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.ZIP_MODE = False
_C.DATA.CACHE_MODE = 'part'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
# -----------------------------------------------------------------------------
# 模型设计
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'bi'
_C.MODEL.NAME = 'biformer_tiny_best.pth'
_C.MODEL.PRETRAIN_CKPT = 'biformer_tiny_best.pth'
_C.MODEL.RESUME = ''
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1
# SWIM TRANSFORMER参数设计
_C.MODEL.BI = CN()
_C.MODEL.BI.PATCH_SIZE = 4
_C.MODEL.BI.IN_CHANS = 3
_C.MODEL.BI.EMBED_DIM = 96
_C.MODEL.BI.DEPTHS = [2, 2, 6, 2]
_C.MODEL.BI.DECODER_DEPTHS = [2, 2, 6, 2]
_C.MODEL.BI.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.BI.WINDOW_SIZE = 7
_C.MODEL.BI.MLP_RATIO = 4.
_C.MODEL.BI.QKV_BIAS = True
_C.MODEL.BI.QK_SCALE = None
_C.MODEL.BI.APE = False
_C.MODEL.BI.PATCH_NORM = True
_C.MODEL.BI.FINAL_UPSAMPLE= "expand_first"
# -----------------------------------------------------------------------------
# 训练设计
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 0
_C.TRAIN.USE_CHECKPOINT = False
# 余弦退火更新学习率，第30次开始进行学习率衰减
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# 定义优化器采用adamw变种其他参数全部使用默认
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# -----------------------------------------------------------------------------
# 数据增强设计
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.25
_C.AUG.REMODE = 'pixel'
_C.AUG.RECOUNT = 1
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.CUTMIX_MINMAX = None
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'
# -----------------------------------------------------------------------------
# 测试设计，中心裁剪
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# 杂项
# -----------------------------------------------------------------------------
_C.AMP_OPT_LEVEL = ''
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0


#(_C.clone,args)
def update_config(config, args):

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)

    return config