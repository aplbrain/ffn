#!/usr/bin/env bash
# GPU-side inference sweep for the labeled Karlupia 256 x 256 x 94 samples.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-third_party/karlupia_inf1}"
LEGACY_INPUT_DIR="${LEGACY_INPUT_DIR:-third_party/karlupia_inf1_legacy}"
RESULTS_DIR="${RESULTS_DIR:-results/karlupia_sweep}"
MODEL_PATH="${MODEL_PATH:-models/karlupia_fused/model.ckpt-2500000}"
DRY_RUN="${DRY_RUN:-0}"

# Override any list with a space-separated environment value. The defaults
# are a diagnostic first-stage sweep, not an expensive final-volume run.
read -r -a SAMPLE_INDICES <<< "${SAMPLE_INDICES:-0 2 4}"
read -r -a AXIS_MODES <<< "${AXIS_MODES:-zyx legacy}"
read -r -a FOV_MODES <<< "${FOV_MODES:-even odd}"
read -r -a MOVE_THRESHOLDS <<< "${MOVE_THRESHOLDS:-0.70 0.80 0.90}"
read -r -a SEGMENT_THRESHOLDS <<< "${SEGMENT_THRESHOLDS:-0.60}"
read -r -a DISCO_THRESHOLDS <<< "${DISCO_THRESHOLDS:--1 0.05}"
read -r -a MIN_SEGMENT_SIZES <<< "${MIN_SEGMENT_SIZES:-500}"

mkdir -p "${RESULTS_DIR}" "${LEGACY_INPUT_DIR}"

for index in "${SAMPLE_INDICES[@]}"; do
  sample="karlupia_inference_256_${index}"
  canonical_input="${INPUT_DIR}/${sample}.h5"
  if [[ ! -f "${canonical_input}" ]]; then
    echo "Missing input: ${canonical_input}" >&2
    exit 1
  fi

  for axis_mode in "${AXIS_MODES[@]}"; do
    case "${axis_mode}" in
      zyx)
        input_file="${canonical_input}"
        ;;
      legacy)
        input_file="${LEGACY_INPUT_DIR}/${sample}.h5"
        if [[ ! -f "${input_file}" ]]; then
          "${PYTHON_BIN}" tools/transpose_h5.py \
            "${canonical_input}" "${input_file}" --axes=2,1,0
        fi
        ;;
      *)
        echo "Unknown axis mode: ${axis_mode}; expected zyx or legacy" >&2
        exit 2
        ;;
    esac

    read -r size_x size_y size_z < <(
      "${PYTHON_BIN}" -c \
        'import h5py,sys; s=h5py.File(sys.argv[1], "r")["raw"].shape; print(s[2],s[1],s[0])' \
        "${input_file}"
    )

    for fov_mode in "${FOV_MODES[@]}"; do
      case "${fov_mode}" in
        even) fov_size='[16, 16, 8]' ;;
        odd) fov_size='[17, 17, 9]' ;;
        *)
          echo "Unknown FOV mode: ${fov_mode}; expected even or odd" >&2
          exit 2
          ;;
      esac

      for move in "${MOVE_THRESHOLDS[@]}"; do
        for segment in "${SEGMENT_THRESHOLDS[@]}"; do
          for disco in "${DISCO_THRESHOLDS[@]}"; do
            for min_size in "${MIN_SEGMENT_SIZES[@]}"; do
              setting="axis-${axis_mode}_fov-${fov_mode}_move-${move}_seg-${segment}_disco-${disco}_min-${min_size}"
              output_path="${RESULTS_DIR}/${sample}/${setting}"
              config_path="${output_path}/inference.pbtxt"
              result_path="${output_path}/0/0/seg-0_0_0.npz"
              mkdir -p "${output_path}"

              if [[ -f "${result_path}" ]]; then
                echo "Skipping completed result: ${result_path}"
                continue
              fi

              cat > "${config_path}" <<EOF
image {
  hdf5: "${input_file}:raw"
}
image_mean: 138.96332215022815
image_stddev: 51.447862638637915
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_checkpoint_path: "${MODEL_PATH}"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 12, \"fov_size\": ${fov_size}, \"deltas\": [8, 8, 4]}"
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

              echo "Running ${sample} ${setting}"
              if [[ "${DRY_RUN}" == 1 ]]; then
                echo "DRY RUN bounding box: x=${size_x} y=${size_y} z=${size_z}"
              else
                "${PYTHON_BIN}" run_inference.py \
                  --inference_request="$(<"${config_path}")" \
                  --bounding_box="start { x:0 y:0 z:0 } size { x:${size_x} y:${size_y} z:${size_z} }"
              fi
            done
          done
        done
      done
    done
  done
done

if find "${RESULTS_DIR}" -type f -path '*axis-zyx_*' -name 'seg-*.npz' \
    -print -quit | grep -q .; then
  echo
  echo "Canonical ZYX scores (best mean adapted Rand error first):"
  "${PYTHON_BIN}" tools/evaluate_karlupia.py \
    "${RESULTS_DIR}/**/axis-zyx_*/0/0/seg-*.npz" \
    --group-setting \
    --csv "${RESULTS_DIR}/metrics_zyx.csv"
fi

if find "${RESULTS_DIR}" -type f -path '*axis-legacy_*' -name 'seg-*.npz' \
    -print -quit | grep -q .; then
  echo
  echo "Legacy-axis scores (best mean adapted Rand error first):"
  "${PYTHON_BIN}" tools/evaluate_karlupia.py \
    "${RESULTS_DIR}/**/axis-legacy_*/0/0/seg-*.npz" \
    --prediction-axes=2,1,0 \
    --group-setting \
    --csv "${RESULTS_DIR}/metrics_legacy.csv"
fi
