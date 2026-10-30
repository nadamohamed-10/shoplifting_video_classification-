# AI-Based Shoplifting Detection Using Deep Learning

A video classification project that investigates deep learning approaches for distinguishing between normal shopping behavior and shoplifting behavior using video data.

The project explores two architectures:

- **CNN-BiGRU with Temporal Attention** — a CNN-based spatial feature extractor combined with recurrent temporal modeling.
- **VideoMAE** — a pretrained Transformer-based video model fine-tuned for binary shoplifting classification.

The main objective is to study how spatial and temporal representations can be used to understand human behavior in video sequences and compare different deep learning architectures in terms of classification performance, generalization, and computational efficiency.

---

## 1. Project Overview

Traditional image classification analyzes individual frames, whereas video classification must capture both:

- **Spatial information:** Visual features, objects, and scene details within individual frames.
- **Temporal information:** Changes in movement and behavior across consecutive frames.

This project treats shoplifting detection as a binary video classification problem.

| Class | Description |
|---|---|
| Normal | Videos showing normal shopping behavior |
| Shoplifting | Videos labeled as shoplifting behavior |

The project follows an experimental approach, starting with a CNN-based temporal architecture and extending the investigation using a pretrained video Transformer.

## 2. Project Objectives

- Develop a video classification pipeline for shoplifting detection.
- Explore video dataset quality and class distribution.
- Extract representative frames using uniform temporal sampling.
- Build a CNN-based model for spatial feature extraction and temporal learning.
- Fine-tune a pretrained VideoMAE model for binary classification.
- Select classification thresholds using validation data.
- Compare model performance using classification metrics.
- Investigate model generalization, duplicate videos, and potential data leakage.
- Evaluate computational requirements and inference performance.

## 3. Project Workflow

The overall workflow consists of:

1. Dataset exploration and preparation.
2. Video metadata inspection and quality checks.
3. Duplicate and near-duplicate video analysis.
4. Uniform temporal frame sampling.
5. Video decoding and preprocessing.
6. Training and evaluation of the CNN-BiGRU baseline.
7. Fine-tuning the pretrained VideoMAE model.
8. Validation-based model and threshold selection.
9. Final test-set evaluation.
10. Performance and computational efficiency comparison.
11. Generalization and data quality analysis.

## 4. Dataset and Preprocessing

### 4.1 Dataset Overview

The dataset contains **855 videos**:

| Class | Number of videos |
|---|---:|
| Normal | 531 |
| Shoplifting | 324 |
| **Total** | **855** |

The original videos have a resolution of approximately 704 × 576 pixels.

### 4.2 Uniform Temporal Sampling

Instead of processing every frame in a video, a fixed number of frames is sampled uniformly across the video duration.

For a video containing \(N\) frames and a sampling configuration of \(T\) frames, the center of each temporal segment is used to determine the sampling index:

\[
f_i = \left\lfloor \frac{(i+0.5)N}{T} \right\rfloor
\]

where \(i = 0,1,\ldots,T-1\).

This approach:

- Maintains temporal coverage across the video.
- Reduces computational requirements.
- Produces a consistent number of frames per video.
- Supports videos with different durations.
- Provides deterministic and reproducible sampling.

### 4.3 Temporal Sampling Analysis

Several frame configurations were investigated.

| Frames per video | Average unique frames | Average unique ratio | Relative input volume |
|---|---:|---:|---:|
| 8 | 8 | 1.00 | 0.5× |
| 16 | 16 | 1.00 | 1.0× |
| 24 | 24 | 1.00 | 1.5× |
| 32 | 32 | 1.00 | 2.0× |

The 16-frame configuration was selected as a practical balance between temporal coverage and computational cost. It also matches the evaluated VideoMAE configuration.

### 4.4 Video Decoding and Preprocessing

OpenCV is used for video decoding and frame processing.

The preprocessing pipeline includes:

- Counting video frames.
- Sequential video decoding.
- Selecting requested frame indices.
- Resizing frames.
- Converting BGR images to RGB.
- Supporting multiple frame configurations.
- Verifying frame indices and chronological ordering.
- Handling unavailable frames through decoding fallbacks.

The Week 6 pipeline also investigates multi-configuration decoding, allowing multiple sampling configurations to be extracted through a shared decoding process.

### 4.5 Duplicate and Near-Duplicate Analysis

The dataset is investigated for exact and visually similar videos.

The analysis includes:

- SHA-256 hashing for exact file identification.
- Perceptual hashing for near-duplicate candidate screening.
- Hamming-distance comparison between perceptual hashes.
- Inspection of candidate pairs with matching or conflicting labels.

Perceptual similarity is treated as a screening signal rather than proof that two videos are identical. Candidate pairs require further inspection before being removed or grouped.

## 5. Model Architectures

### 5.1 CNN-BiGRU with Temporal Attention

The baseline architecture combines a pretrained ResNet18 feature extractor with a Bidirectional GRU and temporal attention.

**Architecture workflow:**

Video Frames → ResNet18 → Frame Features → BiGRU → Temporal Attention → Classification Head → Prediction

**ResNet18**
- Extracts spatial features from individual frames.
- Produces a 512-dimensional feature representation per frame.
- Uses pretrained ImageNet weights.

**Bidirectional GRU**
- Learns temporal dependencies between frames.
- Processes the sequence in forward and backward directions.
- Uses a hidden size of 128 per direction, producing 256-dimensional output representations.

**Temporal Attention**
- Assigns importance weights to temporal representations.
- Allows the model to emphasize more informative parts of the video sequence.

**Classification Head**
- Maps the learned representation to a binary classification logit.
- Predicts Normal or Shoplifting.

### 5.2 VideoMAE — Pretrained Video Transformer

Week 6 introduces VideoMAE as an alternative architecture.

VideoMAE is a Transformer-based video model that uses spatiotemporal representations learned through masked video pretraining.

The model uses pretrained weights from:

`MCG-NJU/videomae-base-finetuned-kinetics`

It is fine-tuned for the project's binary classification task.

**Architecture workflow:**

Video Frames → Tubelet Embedding → Transformer Encoder → Classification Head → Prediction

Key concepts:

- **Pretraining:** The model starts with representations learned from a large video dataset.
- **Tubelet Embedding:** Video information is represented through spatiotemporal tokens.
- **Self-Attention:** The Transformer learns relationships between different parts of the video.
- **Fine-tuning:** The pretrained model is adapted to distinguish Normal and Shoplifting videos.

**Evaluated input configuration:**

| Parameter | Configuration |
|---|---|
| Frames | 16 |
| Resolution | 224 × 224 |
| Channels | 3 (RGB) |
| Classification classes | 2 |
| Pretrained checkpoint | VideoMAE Kinetics-400 fine-tuned checkpoint |

The fine-tuning process uses separate learning rates for the backbone and classification head, with selected Transformer layers unfrozen for adaptation.

## 6. Training and Evaluation Strategy

The dataset is separated into three subsets:

| Split | Purpose |
|---|---|
| Training | Model parameter optimization |
| Validation | Model monitoring and configuration selection |
| Testing | Final performance evaluation |

The validation set is used for:

- Monitoring training and validation loss.
- Selecting the best model checkpoint.
- Comparing validation performance.
- Selecting the classification threshold.

The test set is reserved for final evaluation and should not be used to tune model parameters or thresholds.

### 6.1 VideoMAE Training History

VideoMAE was fine-tuned for 15 recorded epochs.

| Measurement | Result | Epoch |
|---|---:|---:|
| Best validation F1 | 89.23% | 12 |
| Best validation loss | 0.2268 | 13 |
| Final training loss | 0.1850 | 15 |
| Final validation loss | 0.2281 | 15 |
| Final validation F1 | 89.06% | 15 |

The highest validation F1 occurred at epoch 12, while the lowest validation loss occurred at epoch 13.

The training and validation losses generally decreased throughout training. Validation performance improved substantially before reaching a plateau during the final epochs.

There is no clear evidence of severe overfitting from the recorded loss curves, although further analysis of training metrics and independent evaluation is still important.

### 6.2 Classification Threshold

For VideoMAE, the previously selected validation-based classification threshold was **0.28**.

The threshold is selected using validation data rather than test labels to avoid test-set leakage.

### 6.3 Evaluation Metrics

The project uses:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC, when probability predictions are available
- Confusion Matrix

These metrics provide a more complete understanding of classification performance than accuracy alone.

## 7. Experimental Results

The following results reflect the latest reported notebook evaluation.

### 7.1 Classification Performance

| Metric | CNN-BiGRU | VideoMAE |
|---|---:|---:|
| Test Accuracy | **98.44%** | 92.97% |
| Precision | 97.22% | 91.78% |
| Recall | 100.00% | 95.71% |
| F1-score | **98.59%** | 93.71% |
| ROC-AUC | 99.75% | Not available |

**Important:** VideoMAE ROC-AUC is not included because the current recorded evaluation did not provide a valid ROC-AUC result. It should be calculated from test labels and predicted probabilities before being added to the comparison.

### 7.2 Classification Results Interpretation

Based on the latest reported test results:

- CNN-BiGRU achieved an F1-score of 98.59%.
- VideoMAE achieved an F1-score of 93.71%.
- CNN-BiGRU achieved higher reported accuracy, precision, recall, and F1-score in these evaluations.
- VideoMAE demonstrated strong classification performance, with a recall of 95.71%.

The results do not establish that CNN-BiGRU is inherently superior to VideoMAE. A fair architectural comparison requires both models to be evaluated on exactly the same test videos, with the same split and compatible evaluation procedures.

### 7.3 Computational Performance

The latest available VideoMAE measurements are:

| Measurement | VideoMAE |
|---|---:|
| Inference time | 31.805 ms/video |
| Estimated FP32 weight size | 344.915 MB |

The CNN-BiGRU inference time and checkpoint size have not yet been verified under the same benchmarking procedure.

A complete efficiency comparison should measure:

- Number of parameters.
- Actual checkpoint size.
- GPU inference latency.
- CPU inference latency.
- Training time per epoch.
- FLOPs or computational complexity.

Latency measurements should use the same hardware, input configuration, batch size, preprocessing procedure, and warm-up strategy.

## 8. Additional Analysis

The experimental notebooks include several supporting investigations:

- Dataset inventory and class distribution.
- Video metadata and decoding reliability checks.
- Temporal sampling verification.
- Duplicate and near-duplicate candidate analysis.
- Training and validation learning curves.
- Confusion matrix inspection.
- Validation-based threshold selection.
- Bootstrap-based comparison of model F1-scores.
- Computational performance benchmarking.
- Generalization and potential data leakage investigation.

### 8.1 Bootstrap Comparison

An earlier exploratory bootstrap analysis reported:

- P(VideoMAE F1 > CNN-BiGRU F1) = 0.878.
- P(Equal F1) = 0.068.

These values belong to that particular comparison run and should not be interpreted as definitive evidence of statistical superiority. They should be recalculated using predictions from the same test examples before being presented alongside the latest results.

## 9. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning and model training |
| Hugging Face Transformers | Pretrained VideoMAE |
| OpenCV | Video decoding and frame processing |
| NumPy | Numerical operations |
| Pandas | Dataset and experiment analysis |
| Matplotlib | Visualization |
| Scikit-learn | Evaluation metrics and classification analysis |
| ResNet18 | Spatial feature extraction |
| BiGRU | Temporal sequence modeling |
| VideoMAE | Transformer-based video representation learning |


## 11. Conclusion

This project explores two deep learning approaches for video-based shoplifting detection.

The CNN-BiGRU architecture combines pretrained CNN-based spatial feature extraction with recurrent temporal modeling and attention, providing a relatively compact approach to video classification.

VideoMAE introduces a pretrained Transformer-based architecture that learns spatiotemporal representations jointly and can be fine-tuned for shoplifting detection.

In the latest reported evaluations, CNN-BiGRU achieved an F1-score of 98.59%, while VideoMAE achieved 93.71%.

These results demonstrate that both approaches can learn useful representations for the task. However, the results should be interpreted in the context of data quality, split consistency, potential leakage, and computational requirements.

The project highlights the importance of evaluating not only classification performance but also generalization, reproducibility, and practical deployment efficiency.

---

**Focus:** Video Classification · Deep Learning · Computer Vision · Temporal Modeling · CNN · BiGRU · Attention Mechanisms · Vision Transformers
