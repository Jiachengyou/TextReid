<<<<<<< HEAD
<div align="center">
<img src="doc/title.jpg" width="300" height="100" alt="图片名称"/>
</div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sequential-end-to-end-network-for-efficient/person-search-on-prw)](https://paperswithcode.com/sota/person-search-on-prw?p=sequential-end-to-end-network-for-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sequential-end-to-end-network-for-efficient/person-search-on-cuhk-sysu)](https://paperswithcode.com/sota/person-search-on-cuhk-sysu?p=sequential-end-to-end-network-for-efficient)

This repository hosts the source code of our paper: [[AAAI 2021]Sequential End-to-end Network for Efficient Person Search](https://arxiv.org/abs/2103.10148). SeqNet achieves the **state-of-the-art** performance on two widely used benchmarks and runs at **11.5 FPS** on a single GPU. You can find a brief Chinese introduction at [zhihu](https://zhuanlan.zhihu.com/p/358152127).

Performance profile:

| Dataset   | mAP  | Top-1 | Model                                                        |
| --------- | ---- | ----- | ------------------------------------------------------------ |
| CUHK-SYSU | 94.8 | 95.7  | [model](https://drive.google.com/file/d/1wKhCHy7uTHx8zxNS62Y1236GNv5TzFzq/view?usp=sharing) |
| PRW       | 47.6 | 87.6  | [model](https://drive.google.com/file/d/1I9OI6-sfVyop_aLDIWaYwd7Z4hD34hwZ/view?usp=sharing) |

The network structure is simple and suitable as baseline:

![SeqNet](doc/net_arch.jpg)

## Installation

Run `pip install -r requirements.txt` in the root directory of the project.


## Quick Start

Let's say `$ROOT` is the root directory.

1. Download [CUHK-SYSU](https://drive.google.com/open?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af) and [PRW](https://goo.gl/2SNesA) datasets, and unzip them to `$ROOT/data`
```
$ROOT/data
├── CUHK-SYSU
└── PRW
```
2. Following the link in the above table, download our pretrained model to anywhere you like, e.g., `$ROOT/exp_cuhk`
3. Run an inference demo by specifing the paths of checkpoint and corresponding configuration file. `python demo.py --cfg $ROOT/exp_cuhk/config.yaml --ckpt $ROOT/exp_cuhk/epoch_19.pth` You can checkout the result in `demo_imgs` directory.

![demo.jpg](./demo_imgs/demo.jpg)

## Training

Pick one configuration file you like in `$ROOT/configs`, and run with it.

```
python train.py --cfg configs/cuhk_sysu.yaml
```

**Note**: At present, our script only supports single GPU training, but distributed training will be also supported in future. By default, the batch size and the learning rate during training are set to 5 and 0.003 respectively, which requires about 28GB of GPU memory. If your GPU cannot provide the required memory, try smaller batch size and learning rate (*performance may degrade*). Specifically, your setting should follow the [*Linear Scaling Rule*](https://arxiv.org/abs/1706.02677): When the minibatch size is multiplied by k, multiply the learning rate by k. For example:

```
python train.py --cfg configs/cuhk_sysu.yaml INPUT.BATCH_SIZE_TRAIN 2 SOLVER.BASE_LR 0.0012
```

**Tip**: If the training process stops unexpectedly, you can resume from the specified checkpoint.

```
python train.py --cfg configs/cuhk_sysu.yaml --resume --ckpt /path/to/your/checkpoint
```

## Test

Suppose the output directory is `$ROOT/exp_cuhk`. Test the trained model:

```
python train.py --cfg $ROOT/exp_cuhk/config.yaml --eval --ckpt $ROOT/exp_cuhk/epoch_19.pth
```

Test with Context Bipartite Graph Matching algorithm:

```
python train.py --cfg $ROOT/exp_cuhk/config.yaml --eval --ckpt $ROOT/exp_cuhk/epoch_19.pth EVAL_USE_CBGM True
```

Test the upper bound of the person search performance by using GT boxes:

```
python train.py --cfg $ROOT/exp_cuhk/config.yaml --eval --ckpt $ROOT/exp_cuhk/epoch_19.pth EVAL_USE_GT True
```

## Pull Request

Pull request is welcomed! Before submitting a PR, **DO NOT** forget to run `./dev/linter.sh` that provides syntax checking and code style optimation.

## Citation

```
@inproceedings{li2021sequential,
  title={Sequential End-to-end Network for Efficient Person Search},
  author={Li, Zhengjia and Miao, Duoqian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2011--2019},
=======
# Text Based Person Search with Limited Data

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text-based-person-search-with-limited-data/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=text-based-person-search-with-limited-data)

This is the codebase for our [BMVC 2021 paper](https://arxiv.org/abs/2110.10807).

Slides and video for the online presentation are now available at [BMVC 2021 virtual conference website](https://www.bmvc2021-virtualconference.com/conference/papers/paper_0044.html).

## Updates
- (10/12/2021) Add download link of trained models.
- (06/12/2021) Code refactor for easy reproduce.
- (20/10/2021) Code released!

## Abstract
Text-based person search (TBPS) aims at retrieving a target person from an image gallery with a descriptive text query.
Solving such a fine-grained cross-modal retrieval task is challenging, which is further hampered by the lack of large-scale datasets.
In this paper, we present a framework with two novel components to handle the problems brought by limited data.
Firstly, to fully utilize the existing small-scale benchmarking datasets for more discriminative feature learning, we introduce a cross-modal momentum contrastive learning framework to enrich the training data for a given mini-batch. Secondly, we propose to transfer knowledge learned from existing coarse-grained large-scale datasets containing image-text pairs from drastically different problem domains to compensate for the lack of TBPS training data. A transfer learning method is designed so that useful information can be transferred despite the large domain gap.  Armed with these components, our method achieves new state of the art on the CUHK-PEDES dataset with significant improvements over the prior art in terms of Rank-1 and mAP.

## Results
![image](https://user-images.githubusercontent.com/37724292/144879635-86ab9c7b-0317-4b42-ac46-a37b06853d18.png)

## Installation
### Setup environment
```bash
conda create -n txtreid-env python=3.7
conda activate txtreid-env
git clone https://github.com/BrandonHanx/TextReID.git
cd TextReID
pip install -r requirements.txt
pre-commit install
```
### Get CUHK-PEDES dataset
- Request the images from [Dr. Shuang Li](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description).
- Download the pre-processed captions we provide from [Google Drive](https://drive.google.com/file/d/1V4d8OjFket5SaQmBVozFFeflNs6f9e1R/view?usp=sharing).
- Organize the dataset as following:
```bash
datasets
└── cuhkpedes
    ├── annotations
    │   ├── test.json
    │   ├── train.json
    │   └── val.json
    ├── clip_vocab_vit.npy
    └── imgs
        ├── cam_a
        ├── cam_b
        ├── CUHK01
        ├── CUHK03
        ├── Market
        ├── test_query
        └── train_query
```

### Download CLIP weights
```bash
mkdir pretrained/clip/
cd pretrained/clip
wget https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
wget https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt
cd -

```

### Train
```bash
python train_net.py \
--config-file configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml \
--use-tensorboard
```
### Inference
```bash
python test_net.py \
--config-file configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml \
--checkpoint-file output/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048/best.pth
```
You can download our trained models (with CLIP RN50 and RN101) from [Google Drive](https://drive.google.com/drive/folders/1MoceVsLiByg3Sg8_9yByGSvR3ru15hJL?usp=sharing).

## TODO
- [ ] Try larger pre-trained CLIP models.
- [ ] Fix the bug of multi-gpu runninng.
- [ ] Add dataloader for [ICFG-PEDES](https://github.com/zifyloo/SSAN).

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@inproceedings{han2021textreid,
  title={Text-Based Person Search with Limited Data},
  author={Han, Xiao and He, Sen and Zhang, Li and Xiang, Tao},
  booktitle={BMVC},
>>>>>>> 0d976ddc32a6cb6d535980fa7e3dc0ac804e0a2c
  year={2021}
}
```
