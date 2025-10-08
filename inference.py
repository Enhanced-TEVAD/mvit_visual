import os
import torch
from natsort import natsorted
from PIL import Image
import numpy as np
from src.preprocess import oversample_data,load_rgb_batch

def run_mvit(model, frequency, frames_dir, batch_size, sample_mode, device='cpu'):
    assert sample_mode in ['oversample', 'center_crop']
    print("batchsize", batch_size)
    chunk_size = 16
    
    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)  # Convert to tensor
        with torch.no_grad():
            b_data = b_data.float()
            # Extract features directly from the model's trunk
            features = model(b_data)
            # Ensure we get the expected dimension
            if hasattr(features, 'shape') and features.shape[-1] != 768:
                print(f"Warning: Feature dimension is {features.shape[-1]}, expected 768")
        return features.cpu().numpy()

    # Get all TIF files in the directory
    rgb_files = natsorted([i for i in os.listdir(frames_dir) if i.endswith('.tif')])
    frame_cnt = len(rgb_files)
    
    if frame_cnt <= chunk_size:
        print(f"Warning: Only {frame_cnt} frames found, need at least {chunk_size+1}")
        # If we have fewer frames than chunk_size, we'll duplicate the last frame
        if frame_cnt > 0:
            last_frame = rgb_files[-1]
            for i in range(chunk_size + 1 - frame_cnt):
                # Create a copy of the last frame with a new name
                src_path = os.path.join(frames_dir, last_frame)
                dst_path = os.path.join(frames_dir, f"pad_{i:04d}.tif")
                # Copy the file using PIL to ensure compatibility
                img = Image.open(src_path)
                img.save(dst_path)
            # Refresh the file list
            rgb_files = natsorted([i for i in os.listdir(frames_dir) if i.endswith('.tif')])
            frame_cnt = len(rgb_files)
        else:
            print(f"Error: No frames found in {frames_dir}")
            return np.array([])
    
    # Cut frames
    clipped_length = frame_cnt - chunk_size
    clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk
    frame_indices = []  # Frames to chunks
    for i in range(clipped_length // frequency + 1):
        frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])
    frame_indices = np.array(frame_indices)
    chunk_num = frame_indices.shape[0]
    batch_num = int(np.ceil(chunk_num / batch_size))  # Chunks to batches
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)
    
    if sample_mode == 'oversample':
        full_features = [[] for i in range(10)]
    else:
        full_features = [[]]

    for batch_id in range(batch_num):
        print(f"Processing batch {batch_id+1}/{batch_num}")
        batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id])
        if sample_mode == 'oversample':
            batch_data_ten_crop = oversample_data(batch_data)
            for i in range(10):
                assert batch_data_ten_crop[i].shape[-2] == 224
                assert batch_data_ten_crop[i].shape[-3] == 224
                temp = forward_batch(batch_data_ten_crop[i])
                full_features[i].append(temp)
        elif sample_mode == 'center_crop':
            batch_data = batch_data[:, :, 16:240, 58:282, :]
            assert batch_data.shape[-2] == 224
            assert batch_data.shape[-3] == 224
            temp = forward_batch(batch_data)
            full_features[0].append(temp)
    
    if all(len(features) == 0 for features in full_features):
        print("No features extracted!")
        return np.array([])
    
    # Process features based on sample mode
    full_features = [np.concatenate(i, axis=0) for i in full_features if len(i) > 0]
    full_features = [np.expand_dims(i, axis=0) for i in full_features]
    full_features = np.concatenate(full_features, axis=0)
    
# Handle different output shapes based on the model
    if full_features.ndim > 3:
        # Print the shape to debug
        print(f"Feature shape before reduction: {full_features.shape}")
        
        # Preserve the feature dimension (768)
        if full_features.ndim == 4:
            # If shape is (10, N, M, 768), we want to get (10, N, 768)
            # Take only the first feature from dimension M
            full_features = full_features[:, :, 0, :]
        elif full_features.ndim == 5:
            # If shape is (10, N, M, K, 768), we want to get (10, N, 768)
            full_features = full_features[:, :, 0, 0, :]
        elif full_features.ndim == 6:
            # If shape is (10, N, M, K, L, 768), we want to get (10, N, 768)
            full_features = full_features[:, :, 0, 0, 0, :]
        
        # Print the shape after reduction
        print(f"Feature shape after reduction: {full_features.shape}")
    
    # Transpose to get the desired shape (N, 10, feature_dim) for oversample mode
    # or (N, 1, feature_dim) for center_crop mode
    full_features = np.array(full_features).transpose([1, 0, 2])
    return full_features