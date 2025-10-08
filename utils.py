import os
import torch
from src.preprocess import preprocess_video
import numpy as np
from PIL import Image


def extract_features(video_path, model,output_path=None):
    """
    Extract features from a video using MViTv2
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted features (optional)
        
    Returns:
        Extracted features from the video
    """
    # Preprocess video
    video_tensor = preprocess_video(video_path)
    
    if video_tensor is None:
        return None
    
    # Extract features
    with torch.no_grad():
        video_tensor = video_tensor
        features = model(video_tensor)
        features = features.cpu().numpy()
    
    # Save features if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        print(f"Saved features to {output_path}")
    
    return features




