#!/usr/bin/env bash
# GPU-side SmartEM inference plus local/GPU scoring against RGB instance labels.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGE_FILE="${IMAGE_FILE:-third_party/smartem/image_volume.h5}"
LABEL_FILE="${LABEL_FILE:-third_party/smartem/label_volume.h5}"
LEGACY_IMAGE_FILE="${LEGACY_IMAGE_FILE:-third_party/smartem/image_volume_legacy.h5}"
RESULTS_DIR="${RESULTS_DIR:-results/smartem}"
MODEL_PATH="${MODEL_PATH:-models/karlupia_fused/model.ckpt-2500000}"
AXIS_MODE="${AXIS_MODE:-legacy}"
SCORE_ONLY="${SCORE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Defaults are the best practical Karlupia settings. Space-separated overrides
# turn this into a focused sweep without changing the script.
read -r -a MOVE_THRESHOLDS <<< "${MOVE_THRESHOLDS:-0.90}"
read -r -a SEGMENT_THRESHOLDS <<< "${SEGMENT_THRESHOLDS:-0.40}"
read -r -a DISCO_THRESHOLDS <<< "${DISCO_THRESHOLDS:--1}"
read -r -a MIN_SEGMENT_SIZES <<< "${MIN_SEGMENT_SIZES:-250}"

IMAGE_MEAN="${IMAGE_MEAN:-138.96332215022815}"
IMAGE_STDDEV="${IMAGE_STDDEV:-51.447862638637915}"
FOV_SIZE="${FOV_SIZE:-[17, 17, 9]}"

score_results() {
  local prediction_axes
  case "${AXIS_MODE}" in
    zyx) prediction_axes='0,1,2' ;;
    legacy) prediction_axes='2,1,0' ;;
    *)
      echo "Unknown AXIS_MODE=${AXIS_MODE}; expected legacy or zyx" >&2
      return 2
      ;;
  esac

  local first_prediction
  first_prediction="$(find "${RESULTS_DIR}" -type f -name 'seg-*.npz' -print -quit 2>/dev/null || true)"
  if [[ -z "${first_prediction}" ]]; then
    echo "No seg-*.npz predictions found under ${RESULTS_DIR}" >&2
    return 1
  fi

  "${PYTHON_BIN}" tools/evaluate_smartem.py \
    "${RESULTS_DIR}/**/seg-*.npz" \
    --label-file "${LABEL_FILE}" \
    --prediction-axes "${prediction_axes}" \
    --csv "${RESULTS_DIR}/metrics.csv"
  echo "Wrote ${RESULTS_DIR}/metrics.csv"
}

if [[ ! -f "${LABEL_FILE}" ]]; then
  echo "Missing label volume: ${LABEL_FILE}" >&2
  exit 1
fi

if [[ "${SCORE_ONLY}" == 1 ]]; then
  score_results
  exit
fi

if [[ ! -f "${IMAGE_FILE}" ]]; then
  echo "Missing image volume: ${IMAGE_FILE}" >&2
  exit 1
fi

case "${AXIS_MODE}" in
  zyx)
    inference_image="${IMAGE_FILE}"
    ;;
  legacy)
    inference_image="${LEGACY_IMAGE_FILE}"
    if [[ ! -f "${inference_image}" ]]; then
      if [[ "${DRY_RUN}" == 1 ]]; then
        echo "DRY RUN would create ${inference_image} with axes 2,1,0"
      else
        mkdir -p "$(dirname "${inference_image}")"
        "${PYTHON_BIN}" tools/transpose_h5.py \
          "${IMAGE_FILE}" "${inference_image}" \
          --axes=2,1,0 --datasets=raw
      fi
    fi
    ;;
  *)
    echo "Unknown AXIS_MODE=${AXIS_MODE}; expected legacy or zyx" >&2
    exit 2
    ;;
esac

if [[ -f "${inference_image}" ]]; then
  read -r size_x size_y size_z < <(
    "${PYTHON_BIN}" -c \
      'import h5py,sys; s=h5py.File(sys.argv[1], "r")["raw"].shape; print(s[2], s[1], s[0])' \
      "${inference_image}"
  )
else
  # The only missing-file case is a legacy DRY_RUN. Reverse canonical ZYX.
  read -r size_x size_y size_z < <(
    "${PYTHON_BIN}" -c \
      'import h5py,sys; s=h5py.File(sys.argv[1], "r")["raw"].shape; print(s[0], s[1], s[2])' \
      "${IMAGE_FILE}"
  )
fi

mkdir -p "${RESULTS_DIR}"

for move in "${MOVE_THRESHOLDS[@]}"; do
  for segment in "${SEGMENT_THRESHOLDS[@]}"; do
    for disco in "${DISCO_THRESHOLDS[@]}"; do
      for min_size in "${MIN_SEGMENT_SIZES[@]}"; do
        setting="axis-${AXIS_MODE}_move-${move}_seg-${segment}_disco-${disco}_min-${min_size}"
        output_path="${RESULTS_DIR}/${setting}"
        config_path="${output_path}/inference.pbtxt"
        result_path="${output_path}/0/0/seg-0_0_0.npz"
        mkdir -p "${output_path}"

        if [[ -f "${result_path}" ]]; then
          echo "Skipping completed result: ${result_path}"
          continue
        fi

        cat > "${config_path}" <<EOF
image {
  hdf5: "${inference_image}:raw"
}
image_mean: ${IMAGE_MEAN}
image_stddev: ${IMAGE_STDDEV}
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_checkpoint_path: "${MODEL_PATH}"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 12, \"fov_size\": ${FOV_SIZE}, \"deltas\": [8, 8, 4]}"
segmentation_output_dir: "${output_path}"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: ${move}
  disco_seed_threshold: ${disco}
  min_boundary_dist { x: 1 y: 1 z: 1 }
  segment_threshold: ${segment}
  min_segment_size: ${min_size}
}
EOF

        echo "Running ${setting} on bounding box x=${size_x} y=${size_y} z=${size_z}"
        if [[ "${DRY_RUN}" != 1 ]]; then
          "${PYTHON_BIN}" run_inference.py \
            --inference_request="$(<"${config_path}")" \
            --bounding_box="start { x:0 y:0 z:0 } size { x:${size_x} y:${size_y} z:${size_z} }"
        fi
      done
    done
  done
done

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY RUN complete; no inference or scoring was performed."
else
  score_results
fi
