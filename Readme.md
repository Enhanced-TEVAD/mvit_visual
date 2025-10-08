# MVitV2 Video Features Extraction

A Python implementation for extracting visual features from videos using the MVitV2 (Multiscale Vision Transformer) model. This project is designed to process video datasets and extract high-dimensional feature representations suitable for downstream tasks like anomaly detection, action recognition, and video analysis.

## Overview
This project uses PyTorch's pre-trained MVitV2-S model to extract 768-dimensional feature vectors from video frames. It supports batch processing of multiple videos and offers flexible sampling strategies for robust feature extraction.

This project is based on the following repositories:

- [I3D Feature Extraction with ResNet](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet.git)  
- [I3D Feature Extraction with ResNet-50](https://github.com/Guechmed/I3D_Feature_Extraction_resnet_50)

> ✅ The backbone has been replaced with **`MViTv2`** in this version.


## Features

- **Pre-trained MVitV2-S Model**: Uses TorchVision's MViT_V2_S with default ImageNet weights.

- **Batch Processing**: Efficiently processes multiple videos in a directory.

- **Flexible Sampling Modes**

   - `oversample`: 10-crop augmentation (4 corners + center + horizontal flips)

   - `center_crop`: Simple center cropping for faster processing


- **Configurable Parameters:** Adjustable frame sampling frequency and batch sizes

- **Frame Padding:** Automatically handles videos with insufficient frames

## Outputs
The outputs of the model (`MviTv2`) will be represented in vector space and saved in `.npy` files of `768-dim` like that:

  - **(N, 10, 768)** for `oversample mode`
  - **(N, 1, 768)** for `center_crop mode`

  where :

- `N`: represent the number of chunks (number of temporal samples of the video)

- `10` & `1` in the second dimension represent the number of crops (10 for `oversample mode` and 1 for `center_crop mode`)

- `768` in the third dimension represent the number of features.

## Installation

```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py --datasetpath /path/to/videos --outputpath ./output
```