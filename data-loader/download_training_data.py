import numpy as np
import h5py
from cloudvolume import CloudVolume
import os

# ---------------------- Configuration ----------------------

# Replace these with your actual S3 paths
IMAGE_PRECOMP_PATH = 's3://mambo-datalake/connects49a/vsvi2precomputed/inttest'
LABEL_PRECOMP_PATH = 'https://rhoana.rc.fas.harvard.edu/ng/neha_1mm_roi1_seg0419'

# Desired bounding box in XYZ
CROP_SIZE_XYZ = [1536, 1536, 94]  # in voxels

# Define the start of your bounding box (e.g., (0, 0, 0) or custom offset)
BBOX_START_XYZ = [54561//2, 22691//2, 0]

# Output HDF5 paths
OUTPUT_IMAGE_H5 = 'image_volume.h5'
OUTPUT_LABEL_H5 = 'label_volume.h5'

# Dataset names inside HDF5 files
H5_IMAGE_DATASET = 'raw'
H5_LABEL_DATASET = 'labels'

# Uint16 Clamping values
UINT16_MIN = 0
UINT16_MAX = 52221
# ---------------------- Load Volumes ----------------------

def load_cloudvolume_data(path, bbox_start_xyz, crop_size_xyz, mip=0):
    vol = CloudVolume(path, progress=True, use_https=True, mip=mip)
    bbox_end_xyz = [s + c for s, c in zip(bbox_start_xyz, crop_size_xyz)]

    # CloudVolume slicing: [X, Y, Z]
    start_xyz = bbox_start_xyz
    end_xyz = bbox_end_xyz

    data = vol[start_xyz[0]:end_xyz[0], start_xyz[1]:end_xyz[1], start_xyz[2]:end_xyz[2]]
    return np.squeeze(np.asarray(data))


# Load data
print("Downloading image data...")
image_data = load_cloudvolume_data(IMAGE_PRECOMP_PATH, BBOX_START_XYZ, CROP_SIZE_XYZ, mip=1)

print("Downloading label data...")
label_data = load_cloudvolume_data(LABEL_PRECOMP_PATH, BBOX_START_XYZ, CROP_SIZE_XYZ, mip=0)

# ---------------------- Compute Stats ----------------------

# Clamp and scale image data before casting to uint8
#image_data_clamped = np.clip(image_data, UINT16_MIN, UINT16_MAX)

# Scale to [0, 255]
#image_data_scaled = (image_data_clamped / (UINT16_MAX - UINT16_MIN)) * 255.0

# Cast to uint8
#image_data_uint8 = image_data_scaled.astype(np.uint8)


image_mean = float(np.mean(image_data))
image_stddev = float(np.std(image_data))

print(f"Image mean: {image_mean:.4f}")
print(f"Image stddev: {image_stddev:.4f}")

# ---------------------- Save to HDF5 ----------------------

def save_h5(data, filepath, dataset_name):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset(dataset_name, data=data, compression='gzip')
    print(f"Saved {dataset_name} to {filepath}")

# Save image and label volumes
save_h5(image_data, OUTPUT_IMAGE_H5, H5_IMAGE_DATASET)
save_h5(label_data, OUTPUT_LABEL_H5, H5_LABEL_DATASET)
