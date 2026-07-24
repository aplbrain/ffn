#!/usr/bin/env python
"""Builds a TFRecord file of coordinates for training (optimized)."""

from collections import defaultdict

from absl import app
from absl import flags
from absl import logging

import h5py
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_list('partition_volumes', None,
                  'Partition volumes as '
                  '<volume_name>:<volume_path>:<dataset>')
flags.DEFINE_string('coordinate_output', None,
                    'Path to a TF Record file in which to save the coordinates.')
flags.DEFINE_list('margin', None,
                  '(z, y, x) tuple specifying the number of voxels adjacent '
                  'to the border of the volume to exclude.')

IGNORE_PARTITION = 255


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def unravel_indices_3d(indices, shape):
    """Vectorized unravel_index for 3D volumes."""
    return np.stack(np.unravel_index(indices, shape), axis=-1)


def main(argv):
    del argv  # Unused.

    totals = defaultdict(int)
    indices_by_partition = defaultdict(list)

    vol_labels = []
    vol_shapes = []
    mz, my, mx = [int(x) for x in FLAGS.margin]

    logging.info("Loading partition volumes...")
    for i, partvol in enumerate(FLAGS.partition_volumes):
        name, path, dataset = partvol.split(':')
        with h5py.File(path, 'r') as f:
            partitions = f[dataset][mz:-mz, my:-my, mx:-mx]
            vol_shapes.append(partitions.shape)
            vol_labels.append(name)

            uniques, counts = np.unique(partitions, return_counts=True)
            for val, cnt in zip(uniques, counts):
                if val == IGNORE_PARTITION:
                    continue
                totals[val] += cnt
                flat_idxs = np.flatnonzero(partitions == val)
                indices_by_partition[val].extend([(i, idx) for idx in flat_idxs])

    logging.info('Partition counts:')
    for k, v in totals.items():
        logging.info(' %d: %d', k, v)

    logging.info('Resampling and shuffling coordinates...')
    max_count = max(totals.values())
    all_indices = np.concatenate(
        [np.resize(np.random.permutation(v), (max_count, 2)) for v in indices_by_partition.values()],
        axis=0)
    np.random.shuffle(all_indices)

    # Group indices by volume index for efficient vectorized coordinate lookup
    logging.info("Grouping coordinates by volume...")
    grouped_by_volume = defaultdict(list)
    for vol_idx, flat_idx in all_indices:
        grouped_by_volume[vol_idx].append(flat_idx)

    logging.info("Saving coordinates (batched write)...")
    record_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(FLAGS.coordinate_output, options=record_options)

    batch = []
    batch_size = 1000
    total_written = 0

    for vol_idx, flat_indices in grouped_by_volume.items():
        flat_indices = np.array(flat_indices, dtype=np.int64)
        zyx_coords = unravel_indices_3d(flat_indices, vol_shapes[vol_idx])

        for z, y, x in zyx_coords:
            coord = tf.train.Example(features=tf.train.Features(feature=dict(
                center=_int64_feature([mx + x, my + y, mz + z]),
                label_volume_name=_bytes_feature(vol_labels[vol_idx].encode('utf-8'))
            )))
            batch.append(coord.SerializeToString())

            if len(batch) >= batch_size:
                for rec in batch:
                    writer.write(rec)
                total_written += len(batch)
                batch = []

    # Final flush
    for rec in batch:
        writer.write(rec)
    total_written += len(batch)

    writer.close()
    logging.info(f"Done. Total coordinates written: {total_written}")


if __name__ == '__main__':
    flags.mark_flag_as_required('margin')
    flags.mark_flag_as_required('coordinate_output')
    flags.mark_flag_as_required('partition_volumes')

    app.run(main)
