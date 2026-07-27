#!/usr/bin/env python3
"""Score SmartEM FFN segmentations against its RGB-encoded instance labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import glob
import math
import os
from pathlib import Path
import sys

import h5py
import numpy as np


_PAIR_DTYPE = np.dtype([('truth', np.uint32), ('prediction', np.uint64)])


def _parse_axes(value: str) -> tuple[int, int, int]:
  axes = tuple(int(x) for x in value.split(','))
  if sorted(axes) != [0, 1, 2]:
    raise argparse.ArgumentTypeError('axes must be a permutation of 0,1,2')
  return axes


def _packed_label_shape(dataset: h5py.Dataset) -> tuple[int, int, int]:
  if dataset.ndim == 3:
    return tuple(dataset.shape)
  if dataset.ndim == 4 and dataset.shape[0] == 3:
    return tuple(dataset.shape[1:])
  raise ValueError(
      f'label dataset has shape {dataset.shape}; expected ZYX or 3xZYX'
  )


def _read_label_slab(
    dataset: h5py.Dataset, start: int, end: int
) -> np.ndarray:
  if dataset.ndim == 3:
    return np.asarray(dataset[start:end])

  channels = np.asarray(dataset[:, start:end])
  if not np.issubdtype(channels.dtype, np.integer):
    raise ValueError(f'RGB label channels must be integers, got {channels.dtype}')
  if np.any(channels < 0) or np.any(channels > 255):
    raise ValueError('RGB label channels must be in the range 0..255')
  return (
      channels[0].astype(np.uint32)
      | (channels[1].astype(np.uint32) << 8)
      | (channels[2].astype(np.uint32) << 16)
  )


def _add_counts(target: dict, values: np.ndarray) -> None:
  ids, counts = np.unique(values, return_counts=True)
  for label_id, count in zip(ids, counts):
    target[int(label_id)] += int(count)


def _add_joint_counts(
    target: dict[tuple[int, int], int],
    truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
  pairs = np.empty(truth.size, dtype=_PAIR_DTYPE)
  pairs['truth'] = truth.reshape(-1)
  pairs['prediction'] = prediction.reshape(-1)
  unique, counts = np.unique(pairs, return_counts=True)
  for pair, count in zip(unique, counts):
    key = (int(pair['truth']), int(pair['prediction']))
    target[key] += int(count)


def _table_metrics(
    joint: dict[tuple[int, int], int],
    exclude_prediction_zero: bool,
) -> tuple[float, float, float, float, float, int]:
  entries = [
      (truth_id, prediction_id, count)
      for (truth_id, prediction_id), count in joint.items()
      if not exclude_prediction_zero or prediction_id != 0
  ]
  total = sum(count for _, _, count in entries)
  if total == 0:
    return (math.nan, math.nan, math.nan, math.nan, math.nan, 0)

  truth_counts: dict[int, int] = defaultdict(int)
  prediction_counts: dict[int, int] = defaultdict(int)
  for truth_id, prediction_id, count in entries:
    truth_counts[truth_id] += count
    prediction_counts[prediction_id] += count

  joint_pairs = sum(count * (count - 1) for _, _, count in entries)
  truth_pairs = sum(count * (count - 1) for count in truth_counts.values())
  prediction_pairs = sum(
      count * (count - 1) for count in prediction_counts.values()
  )

  precision = joint_pairs / truth_pairs if truth_pairs else math.nan
  recall = joint_pairs / prediction_pairs if prediction_pairs else math.nan
  denominator = 0.5 * truth_pairs + 0.5 * prediction_pairs
  error = 1.0 - joint_pairs / denominator if denominator else math.nan

  vi_split = 0.0
  vi_merge = 0.0
  for truth_id, prediction_id, count in entries:
    probability = count / total
    vi_split += probability * math.log2(truth_counts[truth_id] / count)
    vi_merge += probability * math.log2(
        prediction_counts[prediction_id] / count
    )

  return error, precision, recall, vi_split, vi_merge, total


def evaluate(
    prediction_path: str,
    label_path: str,
    label_dataset: str,
    prediction_axes: tuple[int, int, int],
    prediction_key: str,
    chunk_depth: int,
) -> dict[str, object]:
  with np.load(prediction_path, allow_pickle=False) as data:
    if prediction_key not in data:
      raise KeyError(
          f'{prediction_path!r} has arrays {data.files}; '
          f'missing {prediction_key!r}'
      )
    prediction = np.transpose(data[prediction_key], prediction_axes)

  joint: dict[tuple[int, int], int] = defaultdict(int)
  prediction_sizes: dict[int, int] = defaultdict(int)

  with h5py.File(label_path, 'r') as data:
    if label_dataset not in data:
      raise KeyError(
          f'{label_path!r} has datasets {list(data)}; '
          f'missing {label_dataset!r}'
      )
    labels = data[label_dataset]
    truth_shape = _packed_label_shape(labels)
    if prediction.shape != truth_shape:
      raise ValueError(
          f'shape mismatch: prediction {prediction.shape}, truth {truth_shape}'
      )

    for start in range(0, truth_shape[0], chunk_depth):
      end = min(truth_shape[0], start + chunk_depth)
      truth_slab = _read_label_slab(labels, start, end)
      prediction_slab = np.asarray(prediction[start:end])

      foreground = truth_slab != 0
      _add_joint_counts(
          joint, truth_slab[foreground], prediction_slab[foreground]
      )
      nonzero_prediction = prediction_slab[prediction_slab != 0]
      if nonzero_prediction.size:
        _add_counts(prediction_sizes, nonzero_prediction)

    truth_voxels = int(np.prod(truth_shape))

  error, precision, recall, vi_split, vi_merge, foreground_count = (
      _table_metrics(joint, exclude_prediction_zero=False)
  )
  (
      covered_error,
      covered_precision,
      covered_recall,
      covered_vi_split,
      covered_vi_merge,
      covered_count,
  ) = _table_metrics(joint, exclude_prediction_zero=True)

  prediction_size_values = np.fromiter(
      prediction_sizes.values(), dtype=np.int64
  )
  path = Path(prediction_path)
  setting = path.parents[2].name if len(path.parents) > 2 else path.parent.name
  truth_ids = len({truth_id for truth_id, _ in joint})

  return {
      'sample': 'smartem',
      'setting': setting,
      'prediction_path': os.path.relpath(prediction_path),
      'truth_foreground_fraction': foreground_count / truth_voxels,
      'truth_coverage': covered_count / foreground_count,
      'truth_ids': truth_ids,
      'prediction_ids': len(prediction_sizes),
      'prediction_size_median': (
          float(np.median(prediction_size_values))
          if prediction_size_values.size
          else 0.0
      ),
      'adapted_rand_error': error,
      'adapted_rand_precision': precision,
      'adapted_rand_recall': recall,
      'vi_split': vi_split,
      'vi_merge': vi_merge,
      'covered_adapted_rand_error': covered_error,
      'covered_adapted_rand_precision': covered_precision,
      'covered_adapted_rand_recall': covered_recall,
      'covered_vi_split': covered_vi_split,
      'covered_vi_merge': covered_vi_merge,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'results_glob', help='glob matching seg-*.npz files (quote in the shell)'
  )
  parser.add_argument(
      '--label-file',
      default='third_party/smartem/label_volume.h5',
      help='SmartEM ground-truth HDF5 file',
  )
  parser.add_argument('--label-dataset', default='labels')
  parser.add_argument('--prediction-key', default='segmentation')
  parser.add_argument(
      '--prediction-axes',
      type=_parse_axes,
      default=(0, 1, 2),
      help='transpose prediction before scoring; legacy-axis FFN: 2,1,0',
  )
  parser.add_argument('--chunk-depth', type=int, default=4)
  parser.add_argument('--csv', help='also write metrics to this CSV')
  args = parser.parse_args()

  if args.chunk_depth < 1:
    parser.error('--chunk-depth must be positive')
  paths = sorted(glob.glob(args.results_glob, recursive=True))
  if not paths:
    parser.error(f'no results matched {args.results_glob!r}')

  rows = [
      evaluate(
          path,
          args.label_file,
          args.label_dataset,
          args.prediction_axes,
          args.prediction_key,
          args.chunk_depth,
      )
      for path in paths
  ]
  rows.sort(
      key=lambda row: (
          not math.isfinite(float(row['adapted_rand_error'])),
          float(row['adapted_rand_error']),
      )
  )

  fieldnames = list(rows[0])
  writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
  writer.writeheader()
  writer.writerows(rows)

  if args.csv:
    output_dir = os.path.dirname(args.csv)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with open(args.csv, 'w', newline='', encoding='utf-8') as output:
      file_writer = csv.DictWriter(output, fieldnames=fieldnames)
      file_writer.writeheader()
      file_writer.writerows(rows)


if __name__ == '__main__':
  main()
