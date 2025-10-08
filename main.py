import os
import numpy as np
from tqdm import tqdm
from src.inference import run_mvit
import argparse
from torchvision import models


def process_video_directory(model, sample_mode,batch_size,frequency,directory_path, output_path=None, device='cpu'):
    # Create output directory if needed
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    
    # Get all subdirectories (each should be a video)
    video_dirs = [d for d in os.listdir(directory_path) 
                 if os.path.isdir(os.path.join(directory_path, d))]
    
    features_dict = {}
    
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_path = os.path.join(directory_path, video_dir)
        print(f"\nProcessing video: {video_dir}")
        
        # Extract features
        #frequency = 16  # Sample every 16 frame
        #batch_size = 20  # Process 4 chunks at a time
        #sample_mode = 'oversample'  # Use center crop for feature extraction
        
        features = run_mvit(model, frequency, video_path, batch_size, sample_mode, device)
        
        if features.size > 0:
            # Store features
            features_dict[video_dir] = features
            
            # Save features if output path is provided
            if output_path:
                feature_path = os.path.join(output_path, f"{video_dir}_features.npy")
                np.save(feature_path, features)
                print(f"Saved features to {feature_path}")
                print(f"Feature shape: {features.shape}")
        else:
            print(f"No features extracted for {video_dir}")
    
    return features_dict


if __name__ == '__main__':
    model = models.video.mvit_v2_s(weights="MViT_V2_S_Weights.DEFAULT")
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, required=True, help="Path to UCSDped2 directory.")
    parser.add_argument('--outputpath', type=str, default="output", help="Path to save extracted features.")
    parser.add_argument('--frequency', type=int, default=16, help="Sampling frequency for frames.")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size for feature extraction.")
    parser.add_argument('--sample_mode', type=str, default="oversample", choices=["oversample", "center_crop"], help="Sampling mode."),
    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "cuda","mps"]),
    args = parser.parse_args()

    process_video_directory(
        model,
        args.sample_mode,
        args.batch_size,
        args.frequecy,
        args.datasetpath,
        args.outputpath,
        args.device
    )