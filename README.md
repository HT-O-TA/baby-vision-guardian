# Facial Recognition & Emotion Analysis

A comprehensive facial processing toolkit using advanced deep learning techniques for face detection, cropping, and emotion analysis.

![Facial Recognition Demo](docs/demo.jpg)

## Features

### Face Detection
- **Multi-Model Support**: Implementation of both MTCNN and BlazeFace algorithms
- **High Performance**: BlazeFace provides real-time detection even on CPU 
- **Visualization Tools**: Side-by-side comparison of original images with detection results

### Face Cropping
- **Batch Processing**: Process entire directories of images
- **Smart Filtering**: Intelligently filters out low-quality detections
- **Preservation**: Maintains high-quality facial features for downstream tasks

### Emotion Analysis
- **Custom Model**: Fine-tuned EfficientNet-B0 with CBAM attention mechanism
- **Binary Classification**: Positive/Negative emotion detection
- **Visual Feedback**: Real-time display of prediction results on images

## Project Structure

```
facial_recognition/
├── crop.py                  # MTCNN-based face cropping
├── test_blazeface.py        # BlazeFace implementation 
├── model2.py                # Emotion classification model training
├── predict_model.py         # Emotion prediction with visualization
├── compare_crop.py          # Visualization tool for face detection
├── best_emotion_model.pt    # Pre-trained emotion model
└── README.md                # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/facial_recognition.git
   cd facial_recognition
   ```

2. Create and activate a virtual environment:
   ```bash
   conda create -n facial python=3.8
   conda activate facial
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision 
   pip install facenet-pytorch
   pip install opencv-python matplotlib pillow tqdm
   pip install tensorflow  # For BlazeFace model
   ```

## Usage

### Face Detection & Cropping with MTCNN

Process a folder of images and extract all faces:

```bash
python crop.py
```

This will:
- Process all images in the `input_faces` directory
- Save cropped faces to `output_faces` directory
- Filter out low-quality detections

### Face Detection with BlazeFace

For single image processing with visual comparison:

```bash
python test_blazeface.py -i path/to/image.jpg -o output_dir
```

Optional arguments:
- `-b`: Use back camera model (for detecting smaller/distant faces)

### Emotion Analysis

Predict emotion on a face image with visual results:

```bash
python predict_model.py path/to/face_image.jpg
```

The result will be displayed on screen with emotion label (POSITIVE/NEGATIVE) and confidence score.

### Visualization Tool

Compare detection results with the original images:

```bash
python compare_crop.py
```

## Training Your Own Emotion Model

1. Prepare a dataset with positive and negative emotion face images
2. Organize them in directories following this structure:
   ```
   classified_emotions/
   ├── positive/
   │   ├── img1.jpg
   │   └── ...
   └── negative/
       ├── img1.jpg
       └── ...
   ```
3. Run the training script:
   ```bash
   python model2.py
   ```
4. The best model will be saved as `best_emotion_model.pt`

## Model Architecture

The emotion analysis model uses:
- **Backbone**: EfficientNet-B0 pre-trained on ImageNet
- **Attention**: CBAM (Convolutional Block Attention Module)
- **Classification Head**: Custom classifier with dropout for regularization
- **Training**: AdamW optimizer with learning rate scheduling and class weighting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) for MTCNN implementation
- [MediaPipe](https://mediapipe.dev/) for BlazeFace algorithm
- [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) for backbone architecture 