# Karlupia FFN inference and fine-tuning

## Diagnosis

The current checkpoint was trained from HDF5 arrays shaped `(1536, 1536, 94)`.
The FFN loaders interpret HDF5 array axes as ZYX, but these arrays are physical
XYZ. The coordinate TFRecords confirm the mismatch: sampled XYZ coordinates
have ranges approximately `x=16..77`, `y=24..1511`, `z=24..1511`. In physical
coordinates those are Z, Y, X.

The previous untransposed inference does not test the checkpoint in its
training orientation. It requested a physical `x=512, y=512, z=94` box from an
array shaped `(512, 512, 94)`. FFN interpreted that array as ZYX, for which the
valid box is `x=94, y=512, z=512`, so much of the requested volume was outside
the image and padded.

For the same reason, `results/karlupia/karlupia_inf` is not a valid full-volume
legacy-axis test: the input array is `(1536, 1536, 94)`, but the output request
used a `(94, 1536, 1536)` ZYX result shape instead of the valid legacy-axis
shape `(1536, 1536, 94)`. Its 97% unlabeled output is largely padding. The
`results/karlupia_inf_hannah/image_volume` result is also not comparable: its
source is uint16 with intensities in the hundreds to tens of thousands, while
the checkpoint was trained on uint8 data normalized around 139.

Other findings:

- Training normalization was `mean=138.9, stddev=51.4`; measured full-volume
  values are `138.96332215022815` and `51.447862638637915`. Current inference
  uses `138/54`.
- Omitting `disco_seed_threshold` gives protobuf value zero. That enables the
  strongest disconnectedness-freezing behavior, explicitly biasing inference
  toward oversegmentation.
- The even `[16,16,8]` FOV puts positive movement faces one voxel outside the
  mask. The local movement workaround is now bounds clipping, which preserves
  current even-FOV behavior and permits a symmetric `[17,17,9]` inference test
  with the same fully convolutional checkpoint weights.
- On the six labeled 256 samples, current predictions cover about 60% of
  ground-truth foreground and contain about 599 segments versus 170 true
  objects on average. Mean adapted Rand error is about 0.80.
- Training summaries continue to improve through step 2.5M, so the final
  checkpoint is the best first checkpoint to tune. Training recall at the end
  is only about 0.305 while precision is about 0.993, consistent with the
  observed conservative, fragmented predictions.
- A 50,000-record sample from `train_coords_tiny.tfrecord` heavily weights the
  two extreme local-occupancy bins. The existing `train_coords_large.tfrecord`
  is nearly uniform over all 14 bins, so the fine-tuning launcher uses it by
  default without regenerating coordinates.

## First: tune the existing checkpoint

On the AWS GPU, from the repository root and with the `ffn` environment active:

```bash
conda activate ffn
bash run_karlupia_sweep.sh
```

This runs three labeled samples in both orientations. The legacy path
transposes each input to the array orientation used during training, requests
the valid bounding box, and transposes predictions back only while scoring.
It sweeps movement thresholds `0.70/0.80/0.90` and disconnectedness settings
`-1/0.05`, tests both the trained even FOV and a symmetric odd FOV, and holds
the segment threshold at `0.60`.

Results and CSV metrics are written under `results/karlupia_sweep`. Rank
settings primarily by low `adapted_rand_error`, then inspect the precision and
recall tradeoff and `truth_coverage`. Do not choose by coverage alone because a
merge-heavy result can cover everything.

After identifying the best axis/movement/disconnectedness combination, refine
the segment threshold and minimum size on all six small samples. For example:

```bash
SAMPLE_INDICES='0 1 2 3 4 5' \
AXIS_MODES='legacy' \
FOV_MODES='odd' \
MOVE_THRESHOLDS='0.70' \
DISCO_THRESHOLDS='0.05' \
SEGMENT_THRESHOLDS='0.50 0.60 0.70' \
MIN_SEGMENT_SIZES='100 500' \
RESULTS_DIR=results/karlupia_sweep_stage2 \
bash run_karlupia_sweep.sh
```

Replace the example axis and thresholds with the first-stage winner. Only then
run a 2512 sample. The sweep script intentionally targets 256 samples.

## If inference tuning is insufficient

The preferred next model is a canonical ZYX fine-tune. It uses the existing
checkpoint, HDF5 files, and coordinate TFRecord. The new training flag lazily
transposes only each requested HDF5 patch and swaps TFRecord coordinates in the
graph, so partitions and coordinate files are not regenerated:

```bash
conda activate ffn
AXIS_MODE=zyx MAX_STEPS=3000000 LEARNING_RATE=0.0001 \
bash launch_finetune_karlupia.sh
```

This initializes `models/karlupia_finetune_zyx` from step 2.5M and adds 500k
steps at a tenfold lower learning rate. It also makes the existing default
in-plane augmentation (`permutable_axes=1,2`) physically correct. Evaluate
intermediate checkpoints (for example 2.6M, 2.75M, and 3.0M) with the ZYX sweep
rather than assuming the last checkpoint is best:

```bash
AXIS_MODES=zyx SAMPLE_INDICES='0 2 4' \
MODEL_PATH=models/karlupia_finetune_zyx/model.ckpt-2750000 \
RESULTS_DIR=results/karlupia_finetune_2750000 \
bash run_karlupia_sweep.sh
```

For a lower-risk control that preserves the flawed training orientation:

```bash
AXIS_MODE=legacy MAX_STEPS=2750000 LEARNING_RATE=0.0001 \
bash launch_finetune_karlupia.sh
```

Keep the original `models/karlupia_fused` checkpoint unchanged. The launcher
copies it into a new training directory before resuming.
