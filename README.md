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

### Clone the repo
```bash
git clone https://github.com/VRU-Accident/VRU-Accident.git
cd VRU-Accident
