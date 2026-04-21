#!/usr/bin/env bash
set -euo pipefail

python run.py \
  --is_training 1 \
  --model_file ./models/model_dual.py \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 338 \
  --enc_in 7 \
  --batch_size 16 \
  --train_epochs 8 \
  --patience 2 \
  --learning_rate 0.0005 \
  --gpu 0

