CONFIG=configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml
CONFIG=configs/cuhkpedes/baseline_gru_rn50_ls_bs128_vit.yaml
CONFIG=configs/cuhkpedes/baseline_gru_cliprn50_ls_bs128_vit.yaml
# CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs64_stride12.yaml
CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs128_stride14.yaml
CONFIG=configs/cuhkpedes/VIT/baseline_textvit_clipvit_ls_bs64_stride16_fine_wd4e-3.yaml
# CONFIG=configs/cuhkpedes/VIT/baseline_textvit_clipvit_ls_bs64_stride16.yaml
OUTDIR=./VIT/share_layer12_bs64_fine_wmask_v1_wd4e-3
# RESUME=./output/VIT/baseline_textvit_clipvit_ls_bs96_stride16_test_4/best.pth
CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file $CONFIG --output-dir $OUTDIR
# RESUNME
# --use-tensorboard