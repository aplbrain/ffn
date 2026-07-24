#!/usr/bin/env python3
"""Transpose 3-D HDF5 datasets without loading a full volume into memory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os

import h5py
import numpy as np


def _parse_axes(value: str) -> tuple[int, int, int]:
  axes = tuple(int(x) for x in value.split(','))
  if sorted(axes) != [0, 1, 2]:
    raise argparse.ArgumentTypeError('axes must be a permutation of 0,1,2')
  return axes


def _copy_attrs(source, target) -> None:
  for key, value in source.attrs.items():
    target.attrs[key] = value


def _transpose_dataset(
    source,
    target,
    axes: Sequence[int],
    memory_mb: int,
) -> None:
  # Block along output axis zero. This bounds RAM and makes every write a
  # contiguous slab for the common 2,1,0 Karlupia transpose.
  source_axis = axes[0]
  other_voxels = int(np.prod(source.shape) // source.shape[source_axis])
  bytes_per_plane = max(1, other_voxels * source.dtype.itemsize)
  planes = max(1, memory_mb * 1024 * 1024 // bytes_per_plane)
  chunk_depth = target.chunks[0]
  if planes >= chunk_depth:
    planes = max(chunk_depth, planes // chunk_depth * chunk_depth)

  for start in range(0, source.shape[source_axis], planes):
    end = min(source.shape[source_axis], start + planes)
    source_key = [slice(None)] * 3
    source_key[source_axis] = slice(start, end)
    block = np.asarray(source[tuple(source_key)])
    output_key = [slice(None)] * 3
    output_key[0] = slice(start, end)
    target[tuple(output_key)] = np.transpose(block, axes)


def transpose_h5(
    source_path: str,
    target_path: str,
    axes: Sequence[int],
    datasets: set[str] | None,
    memory_mb: int,
) -> None:
  with h5py.File(source_path, 'r') as source, h5py.File(target_path, 'w') as target:
    _copy_attrs(source, target)
    for name, obj in source.items():
      if not isinstance(obj, h5py.Dataset):
        raise ValueError(f'groups are not supported: {name!r}')
      if datasets is not None and name not in datasets:
        continue
      if obj.ndim != 3:
        raise ValueError(f'dataset {name!r} is {obj.ndim}-D; expected 3-D')

      shape = tuple(obj.shape[i] for i in axes)
      chunks = (
          min(shape[0], 64),
          min(shape[1], 64),
          min(shape[2], 16),
      )
      output = target.create_dataset(
          name,
          shape=shape,
          dtype=obj.dtype,
          chunks=chunks,
          compression='gzip',
          shuffle=True,
      )
      _copy_attrs(obj, output)
      _transpose_dataset(obj, output, axes, memory_mb)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('source')
  parser.add_argument('target')
  parser.add_argument('--axes', type=_parse_axes, default=(2, 1, 0))
  parser.add_argument('--datasets', help='comma-separated datasets; default: all')
  parser.add_argument('--memory-mb', type=int, default=128)
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  if args.memory_mb < 1:
    parser.error('--memory-mb must be positive')
  if os.path.exists(args.target) and not args.overwrite:
    parser.error(f'target already exists: {args.target}')

  selected = set(args.datasets.split(',')) if args.datasets else None
  transpose_h5(args.source, args.target, args.axes, selected, args.memory_mb)


if __name__ == '__main__':
  main()
