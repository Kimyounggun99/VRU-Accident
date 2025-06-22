# VRU-Accident: A Vision-Language Benchmark for Video QA and Dense Captioning on Crash Understanding of Vulnerable Road Users

## 🛑 Overview

**VRU-Accident** is a large-scale vision-language benchmark designed to evaluate multimodal large language models (MLLMs) in traffic accident scenarios involving **Vulnerable Road Users (VRUs)**, such as pedestrians and cyclists.

We provide:
- **1,000 real-world dashcam crash videos**
- **6,000 multiple-choice VQA pairs** across six safety-critical categories
- **1,000 dense crash scene descriptions**

This benchmark uniquely focuses on understanding the **causal, contextual, and preventive** aspects of crashes involving VRUs.

---

## 📦 Benchmark Summary

| Task              | Samples | Candidate Options | Unique Answers |
|-------------------|---------|-------------------|----------------|
| VQA (6 categories)| 6,000   | 24,000            | ~3,400         |
| Dense Captioning  | 1,000   | -                 | 1,000          |

---

## 🧠 Tasks

### 📌 Video Question Answering (VQA)
Each video includes 6 question-answer pairs with 1 correct answer and 3 contextually relevant counterfactuals.

**VQA Categories:**
- Weather & Light 
-  Traffic Environment  
- Road Configuration  
- Accident Type  
- Accident Cause  
- Accident Prevention Measure


### 📝 Dense Captioning
Each video is paired with a detailed, high-quality description capturing:
- Pedestrian/vehicle appearance
- Environmental context
- Weather condition
- Pedestrian/vehicle dynamics
- Spatial relationship between road users
- Detailed collision description

---

## 🚀 Getting Started

### 🧩 Environment

- Python 3.10
- CUDA 12.4
- PyTorch ≥ 2.2

> Tested on Ubuntu 20.04 with NVIDIA TITAN RTX

---

### ⚙️ Installation

```bash
# Create environment
conda create -n VRU-Accident python=3.10 -y
conda activate VRU-Accident

# Clone repository
git clone https://github.com/Kimyounggun99/VRU-Accident.git
cd VRU-Accident

# Install dependencies
pip install -r requirements.txt


🧠 MLLM Weights Setup
You must manually download the following model weights:

Mobile-VideoGPT (0.5B): [link1]

Mobile-VideoGPT (1.5B): [link2]

Video-XL-Pro (3B): [link3]

Video-XL-2 (7B): [link4]

Place them under the following directory structure:

```
./VRU-Accident/Models/
├── Video_XL_Pro_3B/
├── Video-XL-2/
└── Mobile_VideoGPT/
    ├── Mobile_VideoGPT_05B/
    └── Mobile_VideoGPT_15B/
All other model weights will be downloaded automatically during evaluation.

