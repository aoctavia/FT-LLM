# 🩺 Medical Physics LLM Fine-Tuning  
_Fine-tuning Large Language Models for Radiation Physics & Medical Imaging_  

![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange?logo=googlecolab)  
![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-blue?logo=huggingface)  
![License](https://img.shields.io/badge/License-MIT-green)  

---

## 📌 Overview  

This repository demonstrates **fine-tuning a Large Language Model (LLM)** for the **medical physics domain**, covering topics such as radiation therapy, diagnostic imaging, nuclear medicine, and radiation safety.  

The project is part of my **PhD application portfolio in Physics**, showcasing skills in:  

- 🧪 **Domain expertise** in medical physics concepts  
- 🤖 **Machine learning** (NLP, fine-tuning, LoRA/QLoRA)  
- 📊 **Research methodology** (evaluation, benchmarks, domain-specific testing)  
- 💡 **Innovation**: Adapting state-of-the-art LLMs for interdisciplinary science  

---

## ⚙️ Setup & Environment  

- **Platform**: Google Colab (T4 GPU, 16 GB)  
- **Base Models**: `google/gemma-1.1-7b-it`, `unsloth/llama-3-8b`  
- **Fine-tuning Method**: LoRA / QLoRA (parameter-efficient)  
- **Libraries**: HuggingFace Transformers, PEFT, TRL, Unsloth, Accelerate  

To reproduce:  

```bash
pip install transformers==4.44.2 datasets==3.0.0 peft==0.12.0 trl==0.10.1
pip install bitsandbytes==0.43.3 accelerate==0.33.0
pip install unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git
````

---

## 📂 Dataset

Domain-specific Q\&A pairs curated from:

* 📚 Radiation Oncology NLP Database (ROND)
* 📑 AAPM Guidelines & Medical Physics textbooks
* 🧪 Synthetic problems (half-value layer, photoelectric effect, IMRT, etc.)

Data was formatted in **instruction–response style** for supervised fine-tuning.

---

## 🧠 Training

* **Steps**: 100 (demo run)
* **Batch size**: 1 (gradient accumulation = 4)
* **Learning rate**: 2e-4
* **LoRA rank**: 16
* **Final training loss**: `0.4743`
* **Perplexity**: `5.49`

Monitoring done with **Weights & Biases**:
🔗 [Training Dashboard](https://wandb.ai/auliaoctavvia-authfy/huggingface)

---

## 📈 Evaluation

### 1. Inference Testing

Sample prompts and generated responses:
```
Prompt: What is the photoelectric effect in medical imaging?
Response: The Photovoltaic Effect (photoelectron emission) describes how X-rays interact with matter, producing electromagnetic interactions through energy deposition and subsequent electron ejection from atoms or molecules of interest using high-energy radiation such as x-rays. This phenomenon forms the basis for many diagnostic procedures like radiography, fluoroscopy, and digital tomosynthesis.
--------------------------------------------------------------------------------
Prompt: Calculate the half-value layer for 100 keV X-rays in aluminum.
Response: The Half Value Layer (HVL) of any material refers to the thickness required so that radiation intensity reduces by one half compared with its original value. For X-rays specifically, their interaction involves electromagnetic interactions like scattering and absorption, which depend upon energy, frequency, and material composition.
--------------------------------------------------------------------------------
Prompt: Explain the principles of IMRT planning.
Response: IMRT (Intensity Modulated Radiation Therapy) treatment plans are designed using sophisticated software and take into account patient anatomy, tumor characteristics, dose constraints for surrounding healthy tissues, and overall treatment goals. 
- **Patient Positioning:** Ensures accurate targeting with minimal divergence.  
- **Planning Target Definition:** Tumor regions are defined on CT/MRI scans by oncologists and physicists.  
- **Treatment Planning:** 3D models define precise dose delivery while minimizing risk to nearby organs at risk.  
The plan undergoes review and evaluation before clinical use.
--------------------------------------------------------------------------------
Prompt: What are the main components of a PET scanner?
Response: A PET (Positron Emission Tomography) scanner consists of three main components:  
1) A cyclotron to produce isotopes  
2) An injection system for radiopharmaceuticals  
3) A detector ring equipped with radiation detectors  
--------------------------------------------------------------------------------
Prompt: Describe radiation safety principles in nuclear medicine.
Response: Radiation Safety Principles (RSP) include minimizing patient dose through shielding, optimizing effective doses using low-dose protocols, ensuring safe storage/transport/disposal of radioactive materials, and training healthcare professionals on biological effects of radiation to enable informed clinical decisions.
```

### 2. Domain-Specific MCQ Test

* **Q1. Absorbed dose unit?** → Model answered Sv ❌ (Correct: Gy)
* **Q2. IMRT stands for?** → Model answered correctly ✅
* **Q3. Half-life of I-131?** → Model answered correctly ✅

**Domain accuracy**: `66.7% (2/3)`

---

## 🚀 Deployment

* ✅ Model exported (`final_model/`) with `model_info.json`
* ✅ Can be pushed to **HuggingFace Hub** for public/research use

Usage example:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-hf-username/medical-physics-llm")
model = AutoModelForCausalLM.from_pretrained("your-hf-username/medical-physics-llm")

prompt = "What is the photoelectric effect in medical imaging?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🎯 Relevance for PhD Application

This project demonstrates:

* **Bridging physics and AI**: Applying NLP to radiation physics problems
* **Research readiness**: Data curation, fine-tuning, evaluation, and reporting
* **Interdisciplinary innovation**: AI assistant for medical physics education & clinical decision support
* **Commitment to open science**: Code and model shared for reproducibility

---

## 📌 Next Steps

* Expand dataset (AAPM reports, lecture notes)
* Evaluate with **nDCG\@k / Recall\@k** for retrieval tasks
* Add multimodal input (e.g., CT/MRI images + text)
* Explore clinical deployment with guardrails

---

## 📜 Citation

If you use this project:

```bibtex
@misc{octavia2025medphysllm,
  author       = {Aulia Octavia},
  title        = {Medical Physics LLM Fine-Tuning},
  year         = {2025},
  howpublished = {GitHub},
  url          = {https://github.com/aoctavia/FT-LLM}
}
```

---

## 📧 Contact

👤 **Aulia Octavia**

* Email: [auliaoctavvia@gmail.com](mailto:auliaoctavvia@gmail.com)
* GitHub: [@aoctavia](https://github.com/aoctavia)

---
