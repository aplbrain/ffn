"""Lightweight adapters for NumPy-like training volumes."""

from collections.abc import Sequence

import numpy as np


def xyz_coordinate_axes(array_axes: Sequence[int]) -> tuple[int, int, int]:
  """Returns the XYZ-coordinate permutation matching an array permutation."""
  array_axes = tuple(int(x) for x in array_axes)
  if sorted(array_axes) != [0, 1, 2]:
    raise ValueError(f'array_axes must be a permutation of 0,1,2: {array_axes}')
  return tuple(2 - array_axes[2 - axis] for axis in range(3))


class TransposedNumpyLike:
  """Lazy axis-transposed view of a 3-D NumPy-like volume.

  This is useful for HDF5 datasets that were written in XYZ array order even
  though the FFN input pipeline expects ZYX array order. Only the requested
  patch is read from the source dataset; the full volume is never copied.
  """

  def __init__(self, volume, axes: Sequence[int]):
    if volume.ndim != 3:
      raise ValueError(
          f'TransposedNumpyLike requires a 3-D volume; got {volume.ndim}-D.'
      )

    axes = tuple(int(x) for x in axes)
    if sorted(axes) != list(range(volume.ndim)):
      raise ValueError(
          f'axes must be a permutation of 0,1,2; got {axes!r}.'
      )

    self._volume = volume
    self._axes = axes
    self.ndim = volume.ndim
    self.dtype = volume.dtype
    self.shape = tuple(volume.shape[source_axis] for source_axis in axes)

  def __getitem__(self, key):
    if not isinstance(key, tuple):
      key = (key,)
    if len(key) != self.ndim or not all(isinstance(x, slice) for x in key):
      raise IndexError('TransposedNumpyLike expects exactly three slices.')

    source_key = [slice(None)] * self.ndim
    for output_axis, source_axis in enumerate(self._axes):
      source_key[source_axis] = key[output_axis]

    return np.transpose(np.asarray(self._volume[tuple(source_key)]), self._axes)
