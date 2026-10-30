# AI-Based Shoplifting Detection Using Deep Learning

A video classification project that explores deep learning approaches for distinguishing between normal shopping behavior and shoplifting behavior using video data.

The project investigates two architectures:
- **CNN-BiGRU with Temporal Attention** — the baseline video classification model.
- **VideoMAE** — a pretrained Transformer-based video model fine-tuned for shoplifting detection.

The main objective is to study how spatial and temporal representations can be used to understand human behavior in video sequences and compare different deep learning architectures in terms of classification performance and computational efficiency.

---

## 1. Project Overview

Traditional image classification analyzes individual frames, whereas video classification needs to capture both:

- **Spatial information:** Visual features and objects within individual frames.
- **Temporal information:** Changes in movement and behavior across consecutive frames.

This project treats shoplifting detection as a binary video classification problem:

| Class | Description |
|---|---|
| Normal | Videos showing normal shopping behavior |
| Shoplifting | Videos labeled as shoplifting behavior |

The project follows an experimental approach, starting with a CNN-based temporal architecture and extending the work using a pretrained video Transformer.

## 2. Project Objectives

- Develop a video classification pipeline for shoplifting detection.
- Extract representative frames from videos using uniform temporal sampling.
- Build a CNN-based model for spatial feature extraction and temporal learning.
- Fine-tune a pretrained VideoMAE model for binary classification.
- Compare both architectures using classification and computational metrics.
- Investigate model generalization, data quality, and potential data leakage.

## 3. Project Workflow

The overall workflow consists of:

1. Dataset exploration and preparation.
2. Video frame sampling.
3. Video decoding and preprocessing.
4. Training and evaluation of the CNN-BiGRU baseline.
5. Fine-tuning the pretrained VideoMAE model.
6. Validation-based threshold selection.
7. Test-set evaluation.
8. Model performance and efficiency comparison.
9. Data quality and duplicate checks.

## 4. Data Preprocessing

### 4.1 Uniform Temporal Sampling

Instead of processing every frame in a video, a fixed number of frames is sampled uniformly across the entire video.

This approach:
- Maintains temporal coverage.
- Reduces computational requirements.
- Produces a consistent number of frames per video.
- Supports videos with different durations.

The sampling implementation also includes verification checks for index validity, chronological ordering, and approximately uniform temporal gaps.

### 4.2 Video Decoding

OpenCV is used for video decoding and frame processing.

The preprocessing pipeline includes:
- Counting frames.
- Sequential video decoding.
- Selecting requested frame indices.
- Resizing frames.
- Converting BGR images to RGB.
- Supporting multiple frame configurations and resolutions.
- Retrying decoding when requested frames are unavailable.
- Using the nearest successfully decoded frame as a fallback when necessary.

An important optimization in the Week 6 pipeline is multi-configuration decoding, which allows the required frames to be extracted for different configurations from a shared decoding process.

## 5. Model Architectures

### 5.1 CNN-BiGRU with Temporal Attention

The baseline architecture combines a pretrained ResNet18 feature extractor with a Bidirectional GRU and temporal attention.

**Architecture workflow:**

Video Frames → ResNet18 → Frame Features → BiGRU → Temporal Attention → Classification Head → Prediction

Main components:

**ResNet18**
- Extracts spatial features from individual frames.
- Produces a 512-dimensional feature representation per frame.

**Bidirectional GRU**
- Learns temporal dependencies between frames.
- Processes the sequence in forward and backward directions.
- Uses a hidden size of 128 per direction, producing 256-dimensional output representations.

**Temporal Attention**
- Assigns importance weights to temporal representations.
- Allows the model to emphasize more informative frames.

**Classification Head**
- Maps the learned representation to a binary classification logit.
- Predicts Normal or Shoplifting.

### 5.2 VideoMAE — Pretrained Video Transformer

Week 6 introduces VideoMAE as an alternative architecture.

VideoMAE is a video-based Masked Autoencoder architecture that uses Transformer-based representations to learn spatial and temporal information.

The model uses pretrained weights from Kinetics-400 and is fine-tuned for the project's binary classification task.

**Architecture workflow:**

Video Frames → Tubelet Embedding → Transformer Encoder → Classification Head → Prediction

Key concepts:

- **Pretraining:** The model starts with representations learned from a large video dataset.
- **Tubelet Embedding:** Video information is divided into spatiotemporal tokens.
- **Self-Attention:** The Transformer learns relationships between different parts of the video.
- **Fine-tuning:** The pretrained model is adapted to distinguish Normal and Shoplifting videos.

For the evaluated configuration, the input consists of 16 RGB frames at 224 × 224 resolution.

## 6. Training and Evaluation Strategy

The dataset is separated into three subsets:

| Split | Purpose |
|---|---|
| Training | Model parameter optimization |
| Validation | Model monitoring and configuration selection |
| Testing | Final performance evaluation |

The validation set is used for selecting the classification threshold.

For VideoMAE, the previously selected threshold was **0.28**.

The test set is reserved for final evaluation and should not be used to tune model parameters or thresholds.

### Evaluation Metrics

The project uses:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

These metrics provide a more complete understanding of classification performance than accuracy alone.

## 7. Experimental Results

The following are the previously reported notebook results.

### 7.1 Classification Performance

| Metric | CNN-BiGRU | VideoMAE |
|---|---:|---:|
| Test Accuracy | 96.61% | 99.15% |
| Precision | 100% | 100% |
| Recall | 91.49% | 97.87% |
| F1-score | 95.56% | 98.92% |
| ROC-AUC | 1.000 | 1.000 |

### 7.2 Computational Efficiency

| Metric | CNN-BiGRU | VideoMAE |
|---|---:|---:|
| Parameters | 11.98M | 86.24M |
| Checkpoint Size | 45.79 MB | 329.05 MB |
| GFLOPs | 18.95 | 360.69 |
| GPU Latency | 5.04 ms | 31.76 ms |
| CPU Latency | 137.5 ms | 2042.89 ms |
| Training Time / Epoch | 14.84 s | 65.23 s |

*Latency and training measurements are hardware- and configuration-dependent.*

### 7.3 Results Interpretation

VideoMAE achieved higher reported test accuracy, recall, and F1-score than the CNN-BiGRU baseline.

However, this improvement comes with a higher computational cost:

- More model parameters.
- Larger checkpoint size.
- Higher inference latency.
- Longer training time per epoch.

Therefore, the CNN-BiGRU model may be more suitable for resource-constrained environments, while VideoMAE offers stronger reported classification performance in the current experiment.

**Important evaluation limitation:** The test set contains 118 clips (71 Normal and 47 Shoplifting). The high reported metrics should be interpreted cautiously until duplicate and near-duplicate videos, split leakage, and independent generalization are fully investigated.

## 8. Additional Analysis

The experimental notebook also includes:

- Temporal sampling verification.
- Video decoding reliability checks.
- Duplicate and similarity analysis.
- Confusion matrix inspection.
- Bootstrap-based comparison of model F1-scores.
- Computational performance benchmarking.

The bootstrap analysis previously reported:
- P(VideoMAE F1 > CNN-BiGRU F1) = 0.878.
- P(Equal F1) = 0.068.

These values are exploratory evidence rather than definitive proof of statistical superiority.

## 9. Technologies and Libraries

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning and model training |
| Hugging Face Transformers | Pretrained VideoMAE |
| OpenCV | Video decoding and frame processing |
| NumPy | Numerical operations and array manipulation |
| Matplotlib | Visualization |
| Scikit-learn | Evaluation metrics and classification analysis |
| ResNet18 | Spatial feature extraction |
| GRU | Temporal sequence modeling |

## 10. Repository Structure

The following is a suggested repository organization. Update it to match the actual files in your repository.

```text
shoplifting-detection/
│
├── notebooks/
│   ├── Week_5_CNN_BiGRU.ipynb
│   └── Week_6_VideoMAE.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── video_sampling.py
│   │   └── video_decoding.py
│   │
│   ├── models/
│   │   ├── cnn_bigru.py
│   │   └── videomae.py
│   │
│   └── evaluation/
│       └── metrics.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── checkpoints/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 11. Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd shoplifting-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch version compatible with your CUDA environment.

## 12. Running the Project

The main experiments are organized in Jupyter notebooks.

1. Open the Week 5 notebook to explore the CNN-BiGRU baseline.
2. Open the Week 6 notebook to explore the improved preprocessing pipeline and VideoMAE experiments.
3. Configure the dataset paths and model settings according to your environment.
4. Run the notebook cells sequentially.


## 13. Limitations and Future Improvements

Potential future directions include:

- Expanding the dataset with more diverse video recordings.
- Performing group-based data splitting to prevent recording-level leakage.
- Investigating duplicate and near-duplicate clips across dataset splits.
- Evaluating performance on an independent dataset.
- Studying overfitting through training and validation learning curves.
- Exploring temporal augmentation and regularization.
- Optimizing VideoMAE for lower inference latency.
- Investigating real-time video inference and deployment.

## 14. Conclusion

This project explores two deep learning approaches for video-based shoplifting detection.

The CNN-BiGRU architecture provides a relatively lightweight solution by combining CNN-based spatial feature extraction with recurrent temporal modeling and attention.

VideoMAE introduces a pretrained Transformer-based approach that learns spatiotemporal representations jointly.

The experiments demonstrate a trade-off between reported classification performance and computational efficiency, highlighting the importance of evaluating both predictive quality and deployment requirements.

---

**Focus:** Video Classification · Deep Learning · Computer Vision · Temporal Modeling · Vision Transformers
