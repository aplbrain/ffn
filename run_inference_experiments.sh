#!/usr/bin/env bash
set -e

CSV_PATH="top_left_corners.csv"
INPUT_DIR="third_party/karlupia_inf1"
RESULTS_DIR="results/karlupia_inf_hannah"
TMP_CONFIG="configs/tmp_inference.pbtxt"

# Constants
IMAGE_MEAN=138
IMAGE_STDDEV=54
MODEL_PATH="models/karlupia_fused/model.ckpt-2500000"
MODEL_NAME="convstack_3d.ConvStack3DFFNModel"
MODEL_ARGS='{\"depth\": 12, \"fov_size\": [16, 16, 8], \"deltas\": [8, 8, 4]}'

mkdir -p "${RESULTS_DIR}"

# Initialize index
index=0

# Read CSV line by line, skipping header
tail -n +2 "$CSV_PATH" | while IFS=',' read -r x y z size; do
  # Ensure values are clean
  x=$(echo "$x" | tr -d '\r')
  y=$(echo "$y" | tr -d '\r')
  z=$(echo "$z" | tr -d '\r')
  size=$(echo "$size" | tr -d '\r')

  name="karlupia_inference_${size}_${index}"
  input_file="${INPUT_DIR}/${name}.h5"
  output_path="${RESULTS_DIR}/${name}"

  echo "🧠 Processing: $input_file"

  if [ ! -f "$input_file" ]; then
    echo "⚠️  File not found: $input_file — skipping"
    index=$((index + 1))
    continue
  fi

  mkdir -p "$output_path"

  # Create temporary pbtxt config
  cat <<EOF > "${TMP_CONFIG}"
image {
  hdf5: "${input_file}:raw"
}
image_mean: ${IMAGE_MEAN}
image_stddev: ${IMAGE_STDDEV}
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_checkpoint_path: "${MODEL_PATH}"
model_name: "${MODEL_NAME}"
model_args: "${MODEL_ARGS}"
segmentation_output_dir: "${output_path}"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.9
  min_boundary_dist { x: 1 y: 1 z: 1 }
  segment_threshold: 0.6
  min_segment_size: 500
}
EOF

  # Run inference
  python run_inference.py \
    --inference_request="$(cat ${TMP_CONFIG})" \
    --bounding_box "start { x:0 y:0 z:0 } size { x:${size} y:${size} z:94 }"

  echo "✅ Done: $name"
  echo "--------------------------------------"

  # Increment index
  index=$((index + 1))
done
