#!/usr/bin/env bash
# This script launches a training job for the FFN model using TensorFlow.


python train.py \
  --train_coords=/path/to/train_coords-*.tfrecord \
  --data_volumes=myvol:/path/to/images.h5:raw \
  --label_volumes=myvol:/path/to/labels.h5:labels \
  --model_name=ffn.training.models.default_models.ConvStack3DFFNModel \
  --model_args='{"depth": 12, "fov_size": [33, 33, 17], "deltas": [8, 8, 4]}' \
  --train_dir=/path/to/train_output \
  --batch_size=4 \
  --max_steps=100000 \
  --image_mean=138.9 \
  --image_stddev=51.4 \
  --fov_policy=fixed \
  --threshold=0.9 \
  --seed_pad=0.05 \
  --summary_rate_secs=300
