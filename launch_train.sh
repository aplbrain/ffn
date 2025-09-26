#!/usr/bin/env bash
# This script launches a training job for the FFN model using TensorFlow.


python train.py \
  --train_coords=/home/ubuntu/code/ffn/third_party/karlupia/train_coords_tiny.tfrecord \
  --data_volumes=meirovitch:/home/ubuntu/code/ffn/third_party/karlupia/image_volume.h5:raw \
  --label_volumes=meirovitch:/home/ubuntu/code/ffn/third_party/karlupia/label_volume.h5:labels \
  --model_name=convstack_3d.ConvStack3DFFNModel \
  --model_args='{"depth": 12, "fov_size": [16, 16, 8], "deltas": [8, 8, 4]}' \
  --train_dir=/home/ubuntu/code/ffn/models/ \
  --batch_size=4 \
  --max_steps=2500000 \
  --image_mean=138.9 \
  --image_stddev=51.4 \
  --fov_policy=fixed \
  --threshold=0.9 \
  --seed_pad=0.05 \
  --summary_rate_secs=300
