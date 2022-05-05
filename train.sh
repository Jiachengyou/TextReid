CONFIG=configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml
CONFIG=configs/cuhkpedes/baseline_gru_rn50_ls_bs128_vit.yaml
CONFIG=configs/cuhkpedes/baseline_gru_cliprn50_ls_bs128_vit.yaml
# CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs64_stride12.yaml
CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs128_stride14.yaml
CONFIG=configs/cuhkpedes/VIT/baseline_textvit_clipvit_ls_bs64_stride16_fine.yaml
# CONFIG=configs/cuhkpedes/VIT/baseline_textvit_clipvit_ls_bs64_stride16_onehot.yaml
# CONFIG=configs/cuhkpedes/CLIPVIT/baseline_clipvit_textvit_ls_bs64_stride16.yaml
# CONFIG=configs/cuhkpedes/CLIPVIT/baseline_clipvit_ls_bs64_stride16.yaml
# OUTDIR=./VIT/share_layer12_bs64_seed2_wofine_wd4e-4_fine_newbaseline
OUTDIR=./VIT/share_layer12_bs64_seed0_wofine_wd4e-4_fine_weight_cdcr
# OUTDIR=./CLIPVIT/baseline_clipvit_textvit_ls_bs64_stride16_layer6
# OUTDIR=./CLIPVIT/baseline_clipvit_ls_bs64_stride16_ViT-B-32
# RESUME=./output/VIT/share_layer12_bs64_seed0_wofine_wd4e-4_triplt/best.pth
CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file $CONFIG --output-dir $OUTDIR 
# --resume-from $RESUME
# RESUNME
# --use-tensorboard