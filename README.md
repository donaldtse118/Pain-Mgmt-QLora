# Pain-Mgmt-QLora: Clinical Q&A Model Fine-Tuning for Opioid Dosing Decisions

![Python](https://img.shields.io/badge/python-3.10-blue)  
![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-orange?logo=pytorch)  
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow?logo=huggingface)  
![Status](https://img.shields.io/badge/status-experimental-red)

This project fine-tunes a lightweight language model (e.g., `lmsys/vicuna-7b-v1.5`) to answer clinical yes/no questions about opioid dosing in pain management. It targets GPU-constrained environments (8GB VRAM) using 4-bit quantization, efficient prompt-label alignment with attention masking, and controlled data augmentation. It leverages the public [Q-Pain dataset from PhysioNet](https://physionet.org/content/q-pain/1.0.0/) for patient vignettes, drug information, and dosage references.

---

## 🧪 Problem

How to reliably generate yes/no opioid dosage decisions from synthetic clinical vignettes in a pain management context?

---

## Challenges

- **Tiny, imbalanced dataset:** Approximately 50 samples, mostly low dosage, no high dosage, and few omissions. The dataset was originally designed for LLM bias detection rather than dosing prediction, limiting its direct clinical utility.  
- **Data augmentation:** Required to create more medically plausible samples without compromising clinical validity.  
- **Model selection trade-offs:** Models larger than 7B parameters run too slowly on 8GB GPUs. Needed a base model that’s lightweight yet capable and precise in following instructions, including generating JSON-formatted output.

---

## 🛠️ Techniques

- **4-bit quantization** with `bitsandbytes` enables inference of 7B-parameter models on limited GPU memory.  
- **Prompt + label design:**  
  - Concatenate prompt and label sequences separated by `<eos>` tokens for decoder models.  
  - Use attention masks to exclude prompt and padding tokens from loss calculation to prevent label leakage and focus training on the label tokens.  
- **Data pipeline:**  
  - Synthetic vignette generation with control over pain severity and diagnoses (mild, moderate, severe).  
  - Toggle between augmented, original, or hybrid datasets combining both, with targeted sampling to ensure class balance during training.  
- **Error analysis:**  
  - Weak performance on “Low” dosage class traced to limited real samples; added diverse synthetic low-dosage cases to improve class balance.  
  - Observed periods where synthetic data boosted training performance but caused degradation on real clinical data, highlighting the need for higher-quality, clinically valid synthetic samples.

---

## 🧠 Key Insights

- Quantization is critical for efficient LLM inference on GPU-constrained hardware.  
- Removing diagnosis tokens (`mild`, `moderate`, `severe`) from prompts reduces performance on the augmented dataset but slightly improves accuracy on real clinical data—possibly due to overfitting on synthetic diagnostic features.  
- Dataset balancing significantly improves model handling of underrepresented classes like “Low” dosage, which is essential for clinical decision tasks.  
- Synthetic data quality directly impacts real data performance; poorly controlled augmentation can harm real-world accuracy, necessitating strict validation of synthetic samples.

---

## 🎯 Achievements

- Successfully fine-tuned a lightweight 7B-parameter LLM to produce clinically coherent yes/no opioid dosing decisions within GPU constraints (8GB VRAM).  
- Achieved improved accuracy on both synthetic (augmented) and real clinical (medical) datasets by addressing data imbalance through targeted augmentation.  
- Demonstrated that careful synthetic data curation is essential to maintain or improve performance on real-world clinical data.  
- Enables practical clinical Q&A modeling with quantized models, making deployment on limited hardware feasible.

**Performance metric: Accuracy**

| Model Type | Test Dataset | Score | Improvement vs Baseline |
|:----------:|:------------:|:-----:|:-----------------------:|
| Baseline   | Augmented    | 0.500 | —                       |
| Fine-tuned | Augmented    | 0.925 | +0.425 (+85%)           |
| Baseline   | Medical      | 0.775 | —                       |
| Fine-tuned | Medical      | 0.950 | +0.175 (+22.6%)         |

---

## 📚 Domain Knowledge

For detailed clinical context, opioid pharmacology, and pain management concepts referenced in this project, see [domain_knowledge.md](./domain_knowledge.md). This includes opioid types, dosing rules, and pain severity categorizations.

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