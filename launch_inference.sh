#!/usr/bin/env bash
# This script launches an inference job for the FFN model using TensorFlow.

python run_inference.py   --inference_request="$(cat configs/inference_rivlin2025.pbtxt)"   --bounding_box 'start { x:0 y:0 z:0 } size { x:512 y:512 z:512 }'