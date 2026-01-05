# Lightweight Deep Learning Models for Detection of Neurological Disorders Using EEG Signals

## Project Overview
This project focuses on the development of a lightweight deep learning framework for detecting neurological disorders using electroencephalogram (EEG) signals. The system is designed to achieve a balance between diagnostic accuracy and computational efficiency, enabling deployment on resource-constrained edge devices.

The project targets three major neurological disorders:
- Alzheimer’s Disease
- Parkinson’s Disease
- Epilepsy

## Objectives
- Develop efficient deep learning models for EEG-based neurological disorder detection  
- Reduce model complexity while maintaining acceptable accuracy  
- Evaluate performance in terms of accuracy, inference time, and resource usage  
- Enable feasibility for edge and real-time healthcare applications  

## Key Features
- EEG signal preprocessing and normalization  
- Lightweight model architectures (e.g., compact CNN and hybrid models)  
- Performance evaluation using standard EEG datasets  
- Emphasis on edge deployment constraints such as memory and inference latency  

## Technologies Used
- Python  
- PyTorch / TensorFlow  
- NumPy, SciPy  
- Scikit-learn  
- EEG processing libraries  

## Repository Structure
```text
├── data/              # EEG datasets or dataset loaders
├── preprocessing/     # EEG preprocessing scripts
├── models/            # Deep learning model implementations
├── training/          # Training and evaluation scripts
├── experiments/       # Experiment configurations and results
├── utils/             # Helper functions
└── README.md
