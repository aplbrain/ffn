#!/usr/bin/env bash
python compute_partitions.py \
  --input_volume=third_party/karlupia/label_volume.h5:labels \
  --output_volume=third_party/karlupia/partition_volume_sample.h5:partitions \
  --thresholds=0.1,0.5,0.9 \
  --lom_radius=16,16,8 \
  --min_size=10000

#  --thresholds=0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \