#!/usr/bin/env python3
"""Score Karlupia FFN segmentations against labels embedded in input HDF5s."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
import re
import sys

import h5py
import numpy as np
from skimage.metrics import adapted_rand_error
from skimage.metrics import variation_of_information


_SAMPLE_RE = re.compile(r'(karlupia_inference_\d+_\d+)')


def _parse_axes(value: str) -> tuple[int, int, int]:
  axes = tuple(int(x) for x in value.split(','))
  if sorted(axes) != [0, 1, 2]:
    raise argparse.ArgumentTypeError('axes must be a permutation of 0,1,2')
  return axes


def _sample_name(path: str) -> str:
  match = _SAMPLE_RE.search(path)
  if match is None:
    raise ValueError(f'could not find a Karlupia sample name in {path!r}')
  return match.group(1)


def evaluate(
    prediction_path: str,
    input_dir: str,
    prediction_axes: tuple[int, int, int],
) -> dict[str, object]:
  sample = _sample_name(prediction_path)
  input_path = os.path.join(input_dir, f'{sample}.h5')

  with np.load(prediction_path, allow_pickle=True) as data:
    prediction = np.transpose(data['segmentation'], prediction_axes)
  with h5py.File(input_path, 'r') as data:
    truth = np.asarray(data['labels'])

  if prediction.shape != truth.shape:
    raise ValueError(
        f'shape mismatch for {sample}: prediction {prediction.shape}, '
        f'truth {truth.shape}'
    )

  truth_foreground = truth != 0
  covered = truth_foreground & (prediction != 0)
  foreground_count = int(np.count_nonzero(truth_foreground))
  covered_count = int(np.count_nonzero(covered))

  error, precision, recall = adapted_rand_error(
      truth, prediction, ignore_labels=(0,)
  )
  vi_split, vi_merge = variation_of_information(
      truth, prediction, ignore_labels=(0,)
  )

  covered_error = covered_precision = covered_recall = np.nan
  covered_vi_split = covered_vi_merge = np.nan
  if covered_count:
    covered_error, covered_precision, covered_recall = adapted_rand_error(
        truth[covered], prediction[covered]
    )
    covered_vi_split, covered_vi_merge = variation_of_information(
        truth[covered], prediction[covered]
    )

  truth_ids = np.unique(truth[truth_foreground]).size
  prediction_ids, prediction_sizes = np.unique(
      prediction[prediction != 0], return_counts=True
  )

  return {
      'sample': sample,
      'setting': Path(prediction_path).parents[2].name,
      'prediction_path': os.path.relpath(prediction_path),
      'truth_foreground_fraction': foreground_count / truth.size,
      'truth_coverage': covered_count / foreground_count,
      'truth_ids': truth_ids,
      'prediction_ids': prediction_ids.size,
      'prediction_size_median': (
          float(np.median(prediction_sizes)) if prediction_sizes.size else 0.0
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
      '--input-dir', default='third_party/karlupia_inf1', help='labeled HDF5 dir'
  )
  parser.add_argument(
      '--prediction-axes',
      type=_parse_axes,
      default=(0, 1, 2),
      help='transpose prediction before scoring; legacy-axis FFN: 2,1,0',
  )
  parser.add_argument('--csv', help='also write metrics to this CSV')
  parser.add_argument(
      '--group-setting',
      action='store_true',
      help='average all numeric metrics for every inference setting',
  )
  args = parser.parse_args()

  paths = sorted(glob.glob(args.results_glob, recursive=True))
  if not paths:
    parser.error(f'no results matched {args.results_glob!r}')

  rows = [
      evaluate(path, args.input_dir, args.prediction_axes) for path in paths
  ]
  if args.group_setting:
    grouped = {}
    for row in rows:
      grouped.setdefault(row['setting'], []).append(row)

    metric_names = [
        name
        for name, value in rows[0].items()
        if name not in ('sample', 'setting', 'prediction_path')
        and isinstance(value, (int, float, np.integer, np.floating))
    ]
    rows = [
        {
            'setting': setting,
            'sample_count': len(setting_rows),
            **{
                name: float(np.mean([row[name] for row in setting_rows]))
                for name in metric_names
            },
        }
        for setting, setting_rows in grouped.items()
    ]

  rows.sort(key=lambda row: float(row['adapted_rand_error']))
  fieldnames = list(rows[0])
  writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
  writer.writeheader()
  writer.writerows(rows)

  if args.csv:
    with open(args.csv, 'w', newline='', encoding='utf-8') as output:
      file_writer = csv.DictWriter(output, fieldnames=fieldnames)
      file_writer.writeheader()
      file_writer.writerows(rows)


if __name__ == '__main__':
  main()
