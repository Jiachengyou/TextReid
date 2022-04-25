CONFIG=configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml
CONFIG=configs/cuhkpedes/baseline_gru_rn50_ls_bs128_vit.yaml
CONFIG=configs/cuhkpedes/baseline_gru_cliprn50_ls_bs128_vit.yaml
# CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs64_stride12.yaml
CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs128_stride14.yaml
CONFIG=configs/cuhkpedes/VIT/baseline_textvit_clipvit_ls_bs64_stride16.yaml
# CONFIG=configs/cuhkpedes/VIT/baseline_textvit_clipvit_ls_bs64_stride16.yaml
OUTDIR=./VIT/share_layer12_bs64_seed1_wofine_wd4e-4_extra
# RESUME=./output/VIT/share_layer12_bs64_seed1_wofine_wd4e-4/best.pth
CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file $CONFIG --output-dir $OUTDIR 
# --resume-from $RESUME
# RESUNME
# --use-tensorboard