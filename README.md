# Creative Comic Strip Generator

Deep learning multimodal system for visual storytelling that predicts the next comic panel and generates a matching caption from previous frames and story text.

## Quick Links

- **[Experiments Notebook](experiments.ipynb)** – Full experimental workflow and ablations  
- **[Source Code](src/)** – Model, training, and utility modules  
- **[Results](results/)** – Qualitative examples, sample comics, and loss curves  

## Innovation Summary

**I extend a frame‑prediction baseline with cross‑modal attention and a caption decoder, fusing visual context (previous panels) with BERT text features to generate more coherent future panels and captions.**

The model uses a ResNet encoder, LSTM sequence encoder, cross‑modal attention over visual and textual features, and a U‑Net style decoder with perceptual loss, plus an LSTM caption decoder on top of the fused representation.

## Key Results

Training continued from epoch 10 up to epoch 20 on the StoryReasoning dataset (streaming), with the following training losses:

- Epoch 11: **1.6791**  
- Epoch 12: **1.4275**  
- Epoch 13: **1.7058**  
- Epoch 14: **1.5227**  
- Epoch 15: **1.9970**  
- Epoch 16: **1.8283**  
- Epoch 17: **0.9697** *(lowest observed training loss)*  
- Epoch 18: **2.0261**  
- Epoch 19: **1.2595**  
- Epoch 20: **1.4808**

Summarised table:

| Metric               | Value   |
|----------------------|---------|
| Best epoch           | 17      |
| Best epoch loss      | 0.9697  |
| Final epoch (20) loss| 1.4808  |

Qualitative results in `results/figures/` show that predicted panels remain structurally consistent with the input sequence, while generated captions capture key story entities and actions.

## Most Important Finding

> Cross‑modal attention over both the visual sequence and BERT text embeddings produced more coherent predicted panels and captions than using visual information alone, with the best training loss reached at epoch 17 (0.9697).

## Pre‑Registered Plan vs Implementation

- **Planned:** Train and evaluate the model end‑to‑end in a single Google Colab environment with a GPU, then run final experiments on the same account.  
- **Implemented:** Training epochs 1–20 were run on an initial Colab account with GPU access; due to GPU usage limits this work was continued on a separate university Colab account using the saved checkpoints (`model_epoch_*.pth`, `model_best_multimodal.pth`, `model_final_multimodal.pth`).  
- **Outcome:** Lowest training loss of **0.9697** at epoch 17 and visually more consistent predicted panels plus captions that better align with the narrative.

## Reproducibility Notes

Because of Colab GPU limits, the full 20‑epoch training run was not repeated from scratch on the university account. Instead, the university notebook:

1. Loads the pre‑trained checkpoints saved from the original GPU run.  
2. Re‑runs only the evaluation and qualitative visualization cells.  

This mirrors the final model behaviour without incurring the full GPU cost again and matches the results reported in this README.

## How to Reproduce

1. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/Deepak-dv/comic-strip-generator.git
   cd comic-strip-generator
   pip install -r requirements.txt
