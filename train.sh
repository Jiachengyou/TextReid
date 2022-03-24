CONFIG=configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml
CONFIG=configs/cuhkpedes/baseline_gru_rn50_ls_bs128_vit.yaml
CONFIG=configs/cuhkpedes/baseline_gru_cliprn50_ls_bs128_vit.yaml
CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs64_stride12.yaml
# CONFIG=configs/cuhkpedes/baseline_gru_rn50_ls_bs128_vit_test.yaml
CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file $CONFIG
# --use-tensorboard