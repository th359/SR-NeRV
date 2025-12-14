# SR-NeRV: Improving Embedding Efficiency of Neural Video Representation via Super-Resolution<br>
[<img src="https://img.shields.io/badge/GCCE 2025-lightgray.svg?style=flat&logo=java">](https://www.ieee-gcce.org/2025/index.html)
[![arXiv](https://img.shields.io/badge/arXiv-2409.18497-b31b1b.svg)](https://arxiv.org/abs/2505.00046)

Lightweight super-resolution pretraining ([`sr/`](sr)) and NeRV/HNeRV video representation training ([`NeRV/`](NeRV)).

## Environment Setup
- Recommended: Python 3.8+ and CUDA-enabled PyTorch.
- Install dependencies:
  ```bash
  git clone https://github.com/th359/SR-NeRV.git
  cd SR-NeRV
  conda create -n sr-nerv python=3.8
  conda activate sr-nerv
  pip install -r requirements.txt
  ```

## Train the SR model (DIV2K pretraining)
[`sr/train.py`](sr/train.py) assumes DIV2K HR/LR pairs.

- Example layout when `--data_dir DIV2K`:
- DIV2K can be downloaded from: [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
  ```
  dataset/DIV2K/
    DIV2K_train_HR
    DIV2K_train_LR_bicubic/X2
    DIV2K_valid_HR
    DIV2K_valid_LR_bicubic/X2
  ```
- Quick start: [`sr/run.bash`](sr/run.bash) includes a preset command; edit `data_path`/`out_path` inside the script and run `bash run.bash`.
  ```bash
  cd sr
  bash run.bash
  ```
- Example run:
  ```bash
  cd sr
  python train.py \
    --data_dir /path/to/DIV2K \
    --scale 2 \
    --patch_size 96 \
    --batch_size 16 \
    --epochs 100 \
    --eval_interval 25 \
    --output output_sr \
    --lr 1e-4 \
    --warmup_epochs 5 \
    --lr_milestones 200 400 600 \
    --lr_gamma 0.5
  ```
- Checkpoints (e.g., `best_model.pth`) are saved under `--output` and are used as `--sr_weight` for NeRV/HNeRV training.

## Train NeRV / HNeRV models
Training scripts live in [`NeRV/`](NeRV) ([`train_nerv_all.py`](NeRV/train_nerv_all.py) and `run_*.bash`). Point `sr_weight` to the SR checkpoint from above.

- Example: bunny dataset
  ```bash
  cd NeRV
  # Adjust base_path (dataset root) and sr_path (SR weights) in run_bunny.bash if needed
  bash run_bunny.bash
  ```
- Minimal direct invocation:
  ```bash
  cd NeRV
  python train_nerv_all.py \
    --outf bunny \
    --exp_id bunny \
    --data_path /path/to/bunny \
    --vid bunny \
    --conv_type convnext pshuffel \
    --act gelu \
    --norm none \
    --crop_list 640_1280 \
    --resize_list -1 \
    --loss L2 \
    --enc_strds 5 4 4 2 2 \
    --enc_dim 64_16 \
    --dec_strds 5 4 2 2 2 \
    --ks 0_1_5 \
    --reduce 1.2 \
    --modelsize 1.38 \
    --epochs 300 \
    --eval_freq 300 \
    --lower_width 12 \
    --batchSize 1 \
    --lr 0.001 \
    --scale 2 \
    --sr_weight ../sr/output/best_model.pth \
    --finetune_epoch 6 \
    --dump_images
  ```
- For other datasets, use `run_davis.bash` or `run_mcl.bash` as templates and adjust `data_path` and dataset lists.

## Acknowledgement
Our code is based on [HNeRV](https://github.com/haochen-rye/HNeRV).
## Contact
hayatai17@fuji.waseda.jp
