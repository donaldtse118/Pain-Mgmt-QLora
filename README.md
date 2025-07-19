# Pain-Mgmt-QLora: Clinical Q&A Model Fine-Tuning for Opioid Dosing Decisions

![Python](https://img.shields.io/badge/python-3.10-blue)  
![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-orange?logo=pytorch)  
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow?logo=huggingface)  
![Status](https://img.shields.io/badge/status-experimental-red)

This project fine-tunes a lightweight language model (e.g. `lmsys/vicuna-7b-v1.5`) to answer clinical yes/no questions about opioid dosing in pain management. It targets GPU-constrained environments (8GB VRAM) using 4-bit quantization, efficient prompt-label alignment, and controlled data augmentation. It leverages the public [Q-Pain dataset from PhysioNet](https://physionet.org/content/q-pain/1.0.0/) for patient vignettes, drugs, and dosage references.

---

## 🧪 Problem

How to reliably generate yes/no opioid dosage decisions from synthetic clinical vignettes in a pain management context?

---

## Challenges

- **Tiny, imbalanced dataset:** Original dataset has ~50 samples mostly low dosage, no high dosage, few omissions. Originally designed for LLM bias detection, not dosing prediction.
- **Data augmentation:** Needed to create more medically plausible samples without breaking clinical validity.
- **Model selection trade-offs:** Models >7B parameters run too slow on 8GB GPUs. Needed a base model that’s capable but lightweight and follows instructions precisely, including JSON-formatted output.

---

## 🛠️ Techniques

- **4-bit quantization** with `bitsandbytes` enables 7B model inference on limited GPUs.
- **Prompt + label design:**  
  - Concatenate prompt + label + `<eos>` tokens for decoder models.  
  - Use attention masks to exclude prompt and padding from loss calculation.
- **Data pipeline:**  
  - Synthetic vignette generation with control over pain severity and diagnoses.  
   - Toggle between augmented, original, or a hybrid dataset combining both, with targeted sampling to ensure class balance during training.
- **Error analysis:**  
  - Weak performance on “Low” dosage class traced to limited real samples; added diverse synthetic low-dosage cases to improve class balance.  
  - Noticed a period where synthetic data boosted training performance but caused degradation on real medical data, highlighting the need for higher-quality, clinically valid synthetic samples.
  

---

## 🧠 Key Insights

- Quantization is critical for running LLMs in GPU-constrained setups.
- Removing diagnosis (`mild`, `moderate`, `servere`) tokens hurts augmented dataset performance but slightly boosts accuracy on real clinical data.
- Dataset balancing improves model handling of underrepresented classes like “Low” dosage.
- Synthetic data quality directly impacts real data performance; poorly controlled synthetic augmentation can harm real-world accuracy, necessitating stricter validation of synthetic samples.

---


## 📚 Domain Knowledge

For detailed clinical context, opioid pharmacology, and pain management concepts referenced in this project, see [domain_knowledge.md](./domain_knowledge.md).

---

## 📁 Data Source & License

This project uses the [Q-Pain dataset from PhysioNet](https://physionet.org/content/q-pain/1.0.0/), licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

The dataset is **not included** in this repository and must be downloaded separately. Please refer to the original source for full licensing details.

The code in this repository is licensed under the MIT License.

---

## How to Reproduce

1. Download medical dataset [here](https://physionet.org/content/q-pain/1.0.0/), and extract to `{YOUR_PROJECT_ROOT}/local/data/raw/physionet.org`  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Execute, and you could result printed in console [for example](logs/training_run_2025-07-19.log), and 4 csv files with details
   ```base
   python main.py
   ```