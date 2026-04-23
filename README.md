# Lightweight Deep Learning Models for Detection of Neurological Disorders Using EEG Signals

This repository contains our CO425 final year project on lightweight EEG-based deep learning for neurological disorder detection. The project studies a unified design philosophy across three conditions:

- Epilepsy
- Alzheimer's Disease
- Parkinson's Disease

The latest project summary is also presented in the documentation site at [docs/index.html](docs/index.html).

## Project Overview

Neurological disorder diagnosis often depends on specialist review and resource-intensive workflows. In this project, we investigate whether lightweight EEG models can deliver strong classification performance while remaining practical for edge and real-time healthcare settings.

Our current direction is a TEECNet-inspired error-correction approach:

- A lightweight base model first learns dominant EEG patterns
- The base model is then frozen
- A small Error-Correction Network (ECN), or Feature Correction Network (FCN) in the Alzheimer's pipeline, is trained on the residual errors of the base model
- Final predictions are refined without abandoning the lightweight design goal

This lets us improve accuracy while keeping parameter counts low enough for deployable systems.

## Main Contributions

- Unified lightweight EEG classification framework spanning epilepsy, Alzheimer's disease, and Parkinson's disease
- Two-stage training pipeline with residual error correction inspired by TEECNet
- EEG-specific preprocessing pipelines tailored to each dataset and task
- Comparison against lightweight baseline architectures
- Focus on practical metrics such as accuracy, Macro F1, balanced accuracy, AUC, and parameter budget

## Methodology

### Two-Stage Training Pipeline

1. EEG preprocessing
2. Train a lightweight base network
3. Freeze base model parameters
4. Train an ECN/FCN on residual errors
5. Use base prediction + bounded correction as the final output

### Base Model Design

The project uses lightweight EEG architectures built around compact temporal-spatial feature extraction. Across the different tasks, the implementation includes:

- EEGNet-inspired encoders
- Depthwise-separable convolutions
- Temporal-spatial feature extraction
- Frequency-aware feature branches where applicable
- Small correction modules that refine predictions from frozen base models

### Error-Correction Idea

Instead of scaling model size, we learn a targeted correction for the mistakes the base model makes repeatedly. The key hypothesis is that lightweight models have systematic, learnable blind spots, and a small corrector can recover part of that lost accuracy more efficiently than simply building a larger backbone.

## Datasets and Task-Specific Setup

### Epilepsy Detection

- Primary dataset: CHB-MIT
- Cross-dataset evaluation: Siena Scalp EEG
- Preprocessing highlights:
  - channel pad/trim to 23 channels
  - resampling to 256 Hz
  - per-channel Z-score normalization
  - 4-second windows with 50% overlap
  - strong class imbalance handled with focal loss
- Model direction: EEGNet + ECN

### Alzheimer's Disease Detection

- Dataset: OpenNeuro `ds004504`
- Data characteristics: 19 EEG channels sampled at 500 Hz, using a preprocessed derivative
- Preprocessing highlights:
  - band-pass filtering from 0.5 to 45 Hz
  - A1-A2 mastoid re-referencing
  - Artifact Subspace Reconstruction
  - ICA + ICLabel-based component removal
  - resampling from 500 Hz to 250 Hz
  - 20-second windows with 50% overlap
- Model direction: EEGNet-FCN

### Parkinson's Disease Detection

- Primary dataset: OpenNeuro `ds004584`
- Secondary dataset: OpenNeuro `ds002778`
- Preprocessing highlights:
  - band-pass filtering from 0.5 to 45 Hz
  - 50 Hz notch filtering
  - average re-referencing
  - ICA artifact removal
  - downsampling to 250 Hz
  - 2-second windows with 50% overlap
  - subject-wise train/validation/test splitting
- Model directions:
  - Modified EEGNet
  - Lightweight Temporal-Spatial CNN
  - CNN-TCN
  - ECN variants for all three

## Headline Results

The current documentation page highlights the following results:

- Alzheimer's Disease:
  - 93.26% accuracy
  - 93.12% balanced accuracy
  - 93.21% Macro F1
  - 227,137 total parameters for the EEGNet-FCN pipeline
- Parkinson's Disease:
  - up to 96.42% accuracy with CNN-TCN + ECN
  - up to 97.19% F1 with CNN-TCN + ECN
- Epilepsy:
  - 0.9945 AUC with ECN
  - 21.2% F1 improvement in cross-dataset evaluation
  - reduced false alarms compared with the base model

For Alzheimer's detection, the EEGNet-FCN pipeline outperforms the lightweight baselines listed on the project page, including DSCNN, TinyResNet, TCNLite, MobileNet1D, ShuffleNet1D, and SqueezeNet1D.

## Repository Structure

The repository has evolved beyond the original placeholder structure. The main folders currently used are:

```text
.
|-- README.md
|-- docs/
|   |-- index.html          # Project website / documentation page
|   |-- data/               # Site data assets
|   `-- images/             # Team and project images
|-- code/
|   |-- Alzheimers/         # Reserved space for Alzheimer's code
|   |-- Epilepsy/           # Epilepsy module layout
|   |   |-- data/
|   |   |-- experiments/
|   |   |-- models/
|   |   |-- preprocessing/
|   |   |-- training/
|   |   `-- utils/
|   `-- Parkinsons/         # Parkinson's preprocessing and model notebooks
|       |-- data/
|       |-- experiments/
|       |-- models/
|       |-- preprocessing/
|       `-- training/
`-- Alzheimer's Disease Detection/
    |-- experiments/        # Alzheimer's experiment scripts and metrics
    `-- models/             # Saved Alzheimer's model checkpoints
```

## Current Focus by Disorder

- Epilepsy: EEGNet-based seizure detection with ECN-style refinement and cross-dataset evaluation
- Alzheimer's Disease: EEGNet-FCN pipeline on OpenNeuro `ds004504`
- Parkinson's Disease: comparison of three lightweight base models with ECN enhancement on `ds004584` and `ds002778`

## Team

- Sameera Kumarasinghe - E/20/212 - Epilepsy Detection
- Nuwan Dilshan - E/20/455 - Alzheimer's Disease Detection
- Sachin Dulaj - E/20/456 - Parkinson's Disease Detection

### Supervisors

- Prof. Roshan Ragel
- Mr. Sivaraj Nimishan

## Resources

- Repository: <https://github.com/cepdnaclk/e20-4yp-Lightweight-Deep-Learning-Models-for-Detection-of-Neurological-Disorders-Using-EEG-Signals.git>
- Project page source: [docs/index.html](docs/index.html)
- Department of Computer Engineering: <http://www.ce.pdn.ac.lk/>
- University of Peradeniya: <https://eng.pdn.ac.lk/>

## Limitations and Future Work

The current project page identifies several next steps:

- improve generalization with larger and more diverse EEG datasets
- continue reducing the effect of class imbalance
- evaluate real-time and edge deployment more directly
- test hardware latency and power consumption
- move toward a unified multi-disorder detection pipeline
- explore personalized ECN fine-tuning and wearable EEG deployment
