#!/usr/bin/env bash
python build_coordinates.py \
  --partition_volumes=meirovitch:third_party/karlupia/partition_volume_sample.h5:partitions \
  --coordinate_output=third_party/karlupia/train_coords_parallel.tfrecord \
  --margin=24,24,12
