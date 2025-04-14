# Volleyball Activity Recognition with Rally Prediction

## Overview
 
This project implements a deep learning system for volleyball activity recognition using PyTorch. It predicts individual player actions (e.g., spiking, blocking), group activities (e.g., left-pass, right-spike), and rally outcomes (e.g., which team is likely to win the rally) from video clips. The system leverages a combination of a pre-trained AlexNet backbone, LSTM for temporal modeling, self-attention for group interactions, and a graph convolutional network (GCN) for spatial relationships.
 
The project is designed to process the Volleyball Dataset, which contains annotated frames from video clips of volleyball matches. It supports training, evaluation, and visualization modes, with detailed metrics for performance analysis.

## Features

- **Individual Action Classification**
  - Classifies actions for up to 12 players per frame (e.g., spiking, setting, standing).
  - Uses a pre-trained AlexNet backbone with LSTM for temporal context.

- **Group Activity Recognition**
  - Predicts group activities (e.g., left-pass, right-spike, left-winpoint).
  - Employs self-attention to model player interactions.

- **Rally Outcome Prediction**
  - Estimates the probability of each team winning the rally.
  - Combines neural network predictions with a heuristic based on player actions and positions.

- **Visualization**
  - Generates annotated frames with bounding boxes, action labels, group activity predictions, and rally probabilities.

- **Evaluation Metrics**
  - Reports per-class accuracy for actions and group activities.
  - Includes specialized metrics like filtered soft consistency and group-aligned consistency for rally predictions.

## Document Overview

- `main.py`: Entry point for running the project in train, evaluate, or visualize modes.
- `dataset.py`: Defines the `VolleyballDataset` class for loading and processing video clips and annotations.
- `model.py`: Contains the `RallyPredictor` model and its components (e.g., `PersonActionClassifier`, `GroupActivityClassifier`, `GraphFeatureExtractor`).
- `train.py`: Implements the training pipeline with gradient accumulation, mixed-precision training, and learning rate scheduling.
- `evaluate.py`: Evaluates the model on the test set, reporting detailed metrics.
- `infer.py`: Visualizes predictions on test samples, saving annotated frames.
- `requirements.txt`: Lists project dependencies.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PyTorch 2.0+ (with CUDA 11.7+ for GPU, optional)
- CUDA-compatible GPU (optional, for faster training)
- The Volleyball Dataset (not included in this repository; see [Dataset-Github-Repo](https://github.com/mostafa-saad/deep-activity-rec?tab=readme-ov-file) for details)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ethanwong05/FAI_VolleyballAnalyzer.git
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare the dataset: Download the Volleyball Dataset and place it in the `dataset` directory within the project directory. (see [Dataset-Github-Repo](https://github.com/mostafa-saad/deep-activity-rec?tab=readme-ov-file) for details and [Dataset-Download-Link](https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS)).


### Running the Code

The project supports three modes: `train`, `evaluate`, and `visualize`. Use the following commands:

1. **Train the Model**:
   ```bash
   python main.py --mode train --data_dir dataset --output_dir output --num_epochs 60 --batch_size 16
   ```

2. **Evaluate the Model**:
   ```bash
   python main.py --mode evaluate --data_dir dataset --model_path output/checkpoints/best_model_alexnet.pth
   ```
   
3. **Visualize Predictions**:
   ```bash
   python main.py --mode visualize --data_dir dataset --model_path output/checkpoints/best_model_alexnet.pth --num_samples 10
   ```

## Contributing

Contributions are welcome! Fork the repository, create a branch, and submit a pull request with your changes.

## Acknowledgments

- The Volleyball Dataset by Mostafa S. Ibrahim and Greg Mori ([GitHub](https://github.com/mostafa-saad/deep-activity-rec)).
- [PyTorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/) for deep learning.
- [OpenCV](https://opencv.org/) for image processing.
- [Matplotlib](https://matplotlib.org/) for visualization.
