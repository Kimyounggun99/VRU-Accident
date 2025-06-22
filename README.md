# VRU-Accident: A Vision-Language Benchmark for Video QA and Dense Captioning on Crash Understanding of Vulnerable Road Users

## 🛑 Overview

**VRU-Accident** is a large-scale vision-language benchmark designed to evaluate multimodal large language models (MLLMs) in traffic accident scenarios involving **Vulnerable Road Users (VRUs)**, such as pedestrians and cyclists.

We provide:
- **1,000 real-world dashcam crash videos**
- **6,000 multiple-choice VQA pairs** across six safety-critical categories
- **1,000 dense crash scene descriptions**

This benchmark uniquely focuses on understanding the **causal, contextual, and preventive** aspects of crashes involving VRUs.

---

## 📦 Dataset Summary

| Task              | Samples | Candidate Options | Unique Answers |
|-------------------|---------|-------------------|----------------|
| VQA (6 categories)| 6,000   | 24,000            | ~3,400         |
| Dense Captioning  | 1,000   | -                 | 1,000          |

**VQA Categories:**
1. Weather & Light  
2. Traffic Environment  
3. Road Configuration  
4. Accident Type  
5. Accident Cause  
6. Accident Prevention Measure

---

## 🧠 Tasks

### 📌 Video Question Answering (VQA)
Each video includes 6 question-answer pairs with 1 correct answer and 3 contextually relevant counterfactuals.

### 📝 Dense Captioning
Each video is paired with a detailed, high-quality description capturing:
- Pedestrian/vehicle dynamics
- Environmental context
- Collision sequence

---

## 📊 Benchmark Results (VQA)

| Model                 | AccAVG (%) |
|----------------------|------------|
| Gemini 1.5-Flash     | **66.9**   |
| LLaVA-Video (7B)     | 64.4       |
| InternVL3 (8B)       | 64.4       |
| Qwen2-VL (7B)        | 52.1       |
| Human Expert         | **95.0**   |

> MLLMs show strong performance on visually grounded attributes but struggle with **causal reasoning** and **collision understanding**.

---

## 🚀 Getting Started

### Clone the repo
```bash
git clone https://github.com/VRU-Accident/VRU-Accident.git
cd VRU-Accident
