<<<<<<< HEAD
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/prw.yaml
=======
CONFIG=configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml
CONFIG=configs/cuhkpedes/baseline_gru_rn50_ls_bs128_vit.yaml
CONFIG=configs/cuhkpedes/baseline_gru_cliprn50_ls_bs128_vit.yaml
# CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs64_stride12.yaml
CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs128_stride14.yaml
# CONFIG=configs/cuhkpedes/VIT/baseline_gru_clipvit_ls_bs128_stride14_test.yaml
CONFIG=configs/cuhkpedes/VIT/baseline_textvit_clipvit_ls_bs96_stride16.yaml
OUTDIR=./VIT/baseline_textvit_clipvit_ls_bs96_stride16_layer6_cls
# RESUME=./output/VIT/baseline_textvit_clipvit_ls_bs96_stride16_test_4/best.pth
CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file $CONFIG --output-dir $OUTDIR
# RESUNME
# --use-tensorboard
>>>>>>> 0d976ddc32a6cb6d535980fa7e3dc0ac804e0a2c
