#!/usr/bin/env bash
# Continue Karlupia training without rebuilding partitions or coordinate files.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
AXIS_MODE="${AXIS_MODE:-zyx}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-models/karlupia_fused/model.ckpt-2500000}"
TRAIN_COORDS="${TRAIN_COORDS:-third_party/karlupia/train_coords_large.tfrecord}"
MAX_STEPS="${MAX_STEPS:-3000000}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
FOV_POLICY="${FOV_POLICY:-fixed}"

case "${AXIS_MODE}" in
  zyx)
    TRAIN_DIR="${TRAIN_DIR:-models/karlupia_finetune_zyx}"
    AXIS_ARGS=(--input_volume_axes=2,1,0)
    ;;
  legacy)
    TRAIN_DIR="${TRAIN_DIR:-models/karlupia_finetune_legacy}"
    AXIS_ARGS=()
    ;;
  *)
    echo "Unknown AXIS_MODE=${AXIS_MODE}; expected zyx or legacy" >&2
    exit 2
    ;;
esac

if [[ ! -f "${BASE_CHECKPOINT}.index" || ! -f "${BASE_CHECKPOINT}.data-00000-of-00001" ]]; then
  echo "Missing base checkpoint: ${BASE_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${TRAIN_DIR}"
if [[ ! -f "${TRAIN_DIR}/checkpoint" ]]; then
  if find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing to initialize nonempty directory without a checkpoint: ${TRAIN_DIR}" >&2
    exit 1
  fi

  cp "${BASE_CHECKPOINT}.index" "${TRAIN_DIR}/model.ckpt-2500000.index"
  cp "${BASE_CHECKPOINT}.data-00000-of-00001" \
    "${TRAIN_DIR}/model.ckpt-2500000.data-00000-of-00001"
  if [[ -f "${BASE_CHECKPOINT}.meta" ]]; then
    cp "${BASE_CHECKPOINT}.meta" "${TRAIN_DIR}/model.ckpt-2500000.meta"
  fi
  cat > "${TRAIN_DIR}/checkpoint" <<EOF
model_checkpoint_path: "model.ckpt-2500000"
all_model_checkpoint_paths: "model.ckpt-2500000"
EOF
fi

if [[ "${INIT_ONLY:-0}" == 1 ]]; then
  echo "Initialized ${TRAIN_DIR} from ${BASE_CHECKPOINT}"
  exit 0
fi

echo "Fine-tuning ${AXIS_MODE} model in ${TRAIN_DIR} through step ${MAX_STEPS}"
"${PYTHON_BIN}" train.py \
  --train_coords="${TRAIN_COORDS}" \
  --data_volumes=meirovitch:third_party/karlupia/image_volume.h5:raw \
  --label_volumes=meirovitch:third_party/karlupia/label_volume.h5:labels \
  --model_name=convstack_3d.ConvStack3DFFNModel \
  --model_args='{"depth": 12, "fov_size": [16, 16, 8], "deltas": [8, 8, 4]}' \
  --train_dir="${TRAIN_DIR}" \
  --batch_size=4 \
  --max_steps="${MAX_STEPS}" \
  --learning_rate="${LEARNING_RATE}" \
  --image_mean=138.96332215022815 \
  --image_stddev=51.447862638637915 \
  --fov_policy="${FOV_POLICY}" \
  --threshold=0.9 \
  --seed_pad=0.05 \
  --summary_rate_secs=300 \
  "${AXIS_ARGS[@]}"
