import os
import torch
import torch.nn.functional as F
from torchvision.io import read_video
from PIL import Image
import numpy as np


def preprocess_video(video_path, num_frames=16):
    """
    Load and preprocess a video for feature extraction
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample from the video
        
    Returns:
        Preprocessed video tensor of shape [1, 3, T, H, W]
    """
    try:
        # Read video
        video, _, _ = read_video(video_path, pts_unit='sec')
        
        # If video has no frames, return None
        if video.shape[0] == 0:
            print(f"Warning: Video {video_path} has no frames")
            return None
            
        # Convert to float and normalize to [0, 1]
        video = video.float() / 255.0
        
        # Sample frames uniformly
        num_frames = min(num_frames, video.shape[0])
        indices = torch.linspace(0, video.shape[0] - 1, num_frames).long()
        video = video[indices]
        
        # Resize to 224x224
        video = F.interpolate(video.permute(0, 3, 1, 2), size=(224, 224), 
                             mode='bilinear', align_corners=False)
        
        # Normalize with ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video = (video - mean) / std
        
        # Reshape to [1, 3, T, H, W]
        video = video.permute(1, 0, 2, 3).unsqueeze(0)
        
        return video
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None
    

def load_frame(frame_file):
    data = Image.open(frame_file)
    # Convert grayscale to RGB if needed
    if data.mode != 'RGB':
        data = data.convert('RGB')
    data = data.resize((340, 256), Image.Resampling.LANCZOS)
    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1
    assert(data.max() <= 1.0)
    assert(data.min() >= -1.0)
    return data

def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
    return batch_data



def oversample_data(data):
    data_flip = np.array(data[:, :, :, ::-1, :])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5,
            data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]