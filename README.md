# 🧠 Lightweight Deep Learning Models for Detection of Neurological Disorders Using EEG Signals

📌 **Project Overview**

This repository presents a final-year research project conducted under **CO425**, focusing on the design and evaluation of lightweight deep learning architectures for EEG-based neurological disorder detection.

The study targets three clinically significant disorders:

- **Epilepsy**
- **Alzheimer’s Disease (AD)**
- **Parkinson’s Disease (PD)**

The core objective is to investigate whether computationally efficient models can achieve strong classification performance while maintaining **low model complexity**.

🔗 **Project Website**  
[https://cepdnaclk.github.io/e20-4yp-Lightweight-Deep-Learning-Models-for-Detection-of-Neurological-Disorders-Using-EEG-Signals/](https://cepdnaclk.github.io/e20-4yp-Lightweight-Deep-Learning-Models-for-Detection-of-Neurological-Disorders-Using-EEG-Signals/)

---

## 🧠 Motivation

Neurological disorder diagnosis typically relies on:

- Specialist-dependent interpretation
- Time-intensive clinical workflows
- High computational requirements for automated systems

**Electroencephalography (EEG)** provides a non-invasive and cost-effective modality, but existing deep learning approaches are often too resource-heavy.

This project explores **lightweight, EEG-specific architectures** enhanced with **error-correction mechanisms** to improve predictive performance without significantly increasing model size.

---

## ⚙️ Methodology

The proposed framework adopts a **two-stage training paradigm**:

### Stage 1 — Base Model Learning
- Train a lightweight neural network on EEG signals
- Extract dominant **temporal–spatial features**

### Stage 2 — Error-Correction Learning
- Freeze the trained base model
- Train a compact auxiliary network:
  - **Error-Correction Network (ECN)**
  - **Feature Correction Network (FCN)** *(Alzheimer’s-specific variant)*
- Learn residual errors from base predictions
- Refine outputs via **additive correction**

### Final Prediction
```
Final Output = Base Prediction + Correction
```

---

## 🔬 Research Scope

### 🧩 Epilepsy Detection
- **Primary Dataset:** CHB-MIT
- **Cross-Dataset:** Siena Scalp EEG
- **Architecture:** EEGNet + ECN
- **Focus:** Generalization and false alarm reduction

### 🧩 Alzheimer’s Disease Detection
- **Dataset:** OpenNeuro ds004504
- **Architecture:** EEGNet-FCN
- **Focus:** Feature-level correction

### 🧩 Parkinson’s Disease Detection
- **Datasets:**
  - OpenNeuro ds004584
  - OpenNeuro ds002778
- **Architectures:**
  - Modified EEGNet
  - Lightweight Temporal-Spatial CNN
  - CNN-TCN
  - ECN-enhanced variants

---

## 🧪 Methodological Highlights

- EEG-specific preprocessing pipelines per dataset
- Temporal–spatial feature extraction
- Depthwise separable convolutions *(EEGNet-inspired)*
- Residual error modeling using auxiliary networks
- **Multi-metric evaluation:**
  - Accuracy
  - Macro F1-score
  - Balanced Accuracy
  - AUC
  - Parameter count

---

## 📊 Key Results

### Alzheimer’s Disease
| Metric              | Value          |
|---------------------|----------------|
| Accuracy            | **93.26%**     |
| Balanced Accuracy   | **93.12%**     |
| Macro F1            | **93.21%**     |
| Model Size          | **227,137** parameters |

### Parkinson’s Disease
- **Best Accuracy:** 96.42% (CNN-TCN + ECN)
- **Best F1-score:** 97.19%

### Epilepsy
- **AUC:** 0.9945 (with ECN)
- **Cross-Dataset F1 Improvement:** +21.2%
- Significant reduction in false positives compared to baseline

---

## 📁 Repository Structure

```text
.
├── README.md
├── docs/                         # Project website and documentation
├── code/
│   ├── Alzheimers/
│   ├── Epilepsy/
│   └── Parkinsons/
└── Alzheimer's Disease Detection/
    ├── experiments/
    └── models/
```

---

## 👥 Team

| Member                | Contribution                  |
|-----------------------|-------------------------------|
| **Sameera Kumarasinghe** | Epilepsy Detection           |
| **Nuwan Dilshan**        | Alzheimer’s Disease Detection |
| **Sachin Dulaj**         | Parkinson’s Disease Detection |

---

## 🎓 Supervision

- **Prof. Roshan Ragel**
- **Mr. Sivaraj Nimishan**

---

## 🔗 Resources

- 📦 **Repository:** [GitHub](https://github.com/cepdnaclk/e20-4yp-Lightweight-Deep-Learning-Models-for-Detection-of-Neurological-Disorders-Using-EEG-Signals.git)
- 🌐 **Project Page:** [GitHub Pages](https://cepdnaclk.github.io/e20-4yp-Lightweight-Deep-Learning-Models-for-Detection-of-Neurological-Disorders-Using-EEG-Signals/)
- 🏫 **Department:** [Computer Engineering, University of Peradeniya](http://www.ce.pdn.ac.lk/)
- 🎓 **University:** [Faculty of Engineering, University of Peradeniya](https://eng.pdn.ac.lk/)

---

> **Keywords:** EEG, Deep Learning, Lightweight Models, Neurological Disorders, Epilepsy, Alzheimer’s, Parkinson’s, Error Correction, EEGNet

*This project was developed as part of the final-year undergraduate research (CO425) at the Department of Computer Engineering, University of Peradeniya, Sri Lanka.*
