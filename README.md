```markdown
# Creative Comic Strip Generator

Deep learning multimodal system that predicts the next comic panel and generates a matching caption from previous frames and story text.

## Quick Links

- **[CREATIVE_COMIC_STRIP_GENERATOR Notebook](CREATIVE_COMIC_STRIP_GENERATOR.ipynb)** – Full experimental workflow  
- **[Source Code](src/)** – Model, training, and utilities  
- **[Results](results/)** – Qualitative examples and loss curves  

## Innovation Summary

**I extend a frame‑prediction baseline with cross‑modal attention and a caption decoder, fusing visual context with BERT text features to generate more coherent future panels and captions.**

The model combines a ResNet encoder, LSTM sequence encoder, cross‑modal attention, U‑Net style decoder with perceptual loss, and an LSTM caption decoder.

## Key Results

Training (epochs 11–20) on the StoryReasoning dataset yielded:

| Metric               | Value   |
|----------------------|---------|
| Best epoch           | 17      |
| Best epoch loss      | 0.9697  |
| Final epoch (20) loss| 1.4808  |

Qualitative samples in `results/figures/` show predicted panels that stay consistent with the input sequence and captions that reflect key story entities and actions.

## How to Reproduce

1. Install and set up:

   ```bash
   git clone https://github.com/Deepak-dv/comic-strip-generator.git
   cd comic-strip-generator
   pip install -r requirements.txt
   ```

2. Training (optional, GPU recommended):

   - Open `CREATIVE_COMIC_STRIP_GENERATOR.ipynb`.  
   - Run cells to load the dataset, build the model, and train for 1–20 epochs, saving `model_best_multimodal.pth`.

3. Evaluation / visuals:

   - Ensure `model_best_multimodal.pth` and `model_final_multimodal.pth` are in the paths given in `config.yaml`.  
   - In `CREATIVE_COMIC_STRIP_GENERATOR.ipynb`, run only the data/model cells, checkpoint loading, evaluation, and qualitative visualization cells.

### Using the Modular Code

```python
from src.model import EnhancedFramePredictor, EnhancedMultimodalLoss
from src.utils import StoryDataset, get_default_transform, enhance, denorm, vocab_size
from src.train import build_dataloaders, train_epoch, validate

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader, test_loader = build_dataloaders(
    batch_size=2, sequence_length=3,
    max_train=800, max_val=200, max_test=100
)

model = EnhancedFramePredictor(vocab_size=vocab_size).to(device)
criterion = EnhancedMultimodalLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

## Repository Structure

```text
comic-strip-generator/
├── CREATIVE_COMIC_STRIP_GENERATOR.ipynb
├── config.yaml
├── README.md
├── requirements.txt
├── src/
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── results/
    └── figures/
```

## Plan, GPU Limits, and Reproducibility

- **Plan:** Train and evaluate end‑to‑end in a single Colab GPU environment.  
- **Reality:** Epochs 1–20 were trained on an initial Colab account; due to GPU limits, the university Colab account loads these saved checkpoints and re‑runs only evaluation and visualization.  
- **Outcome:** Best training loss **0.9697** (epoch 17) and improved visual and caption coherence, reproducible by loading the same checkpoints on the university account.
