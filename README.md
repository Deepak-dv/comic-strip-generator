# Comic Strip Generator

Deep learning multimodal system for creative comic strip generation using cross-modal attention and GANs.

## Project Overview

This project implements an AI-powered comic strip generator that predicts the next panel in a comic sequence, including both the visual illustration and the dialog text. The system uses advanced deep learning techniques to understand the narrative flow and generate coherent, creative continuations of comic stories.

## Motivation

Comic strips combine visual storytelling with text in a unique way. Traditional AI models struggle to generate content that maintains coherence across both modalities. This project addresses that challenge by implementing cross-modal attention mechanisms that allow the image and text components to inform each other, resulting in more engaging and contextually appropriate comic panels.

## Key Features

- **Multimodal Learning**: Processes both images and text simultaneously
- **Cross-Modal Attention**: Novel attention mechanism that connects visual and textual features
- **GAN-Based Image Generation**: Creates high-quality, style-consistent comic panel illustrations
- **Sequence Modeling**: Uses temporal models to understand story progression
- **Creative Output**: Generates both images and dialog for the next comic panel

## Architecture

The system consists of several key components:

1. **Visual Encoder**: CNN-based encoder (ResNet/VGG) for extracting image features
2. **Text Encoder**: RNN/LSTM/Transformer for processing dialog and descriptions
3. **Cross-Modal Attention Layer**: Allows bidirectional information flow between image and text representations
4. **Sequence Model**: LSTM/GRU for modeling temporal dependencies across panels
5. **Dual Decoders**:
   - GAN-based image decoder for generating comic panel visuals
   - Autoregressive text decoder for generating dialog

## Dataset

We use the **StoryReasoning Dataset** which contains:
- Sequential image-text pairs designed for grounded story generation
- Rich annotations including objects, locations, actions, and descriptions
- Diverse narrative styles suitable for comic generation

## Innovation

Our main contribution is the **cross-modal attention mechanism** that enables:
- Better alignment between visual content and dialog
- More coherent narrative progression
- Enhanced creative generation compared to baseline models

We also integrate **GAN-based decoders** to produce more realistic and style-consistent comic illustrations.

## Applications

This technology can be used for:
- **Assistive Technology**: Help visually impaired individuals experience comic strips through generated narratives
- **Creative Tools**: Assist artists and writers in brainstorming and visualizing story ideas
- **Educational Content**: Generate educational comics for learning materials
- **Entertainment**: Create personalized comic content

## Project Structure

```
comic-strip-generator/
├── data/                    # Dataset files
├── src/                     # Source code
│   ├── models/             # Model architectures
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── notebooks/              # Jupyter notebooks for experiments
├── results/                # Generated outputs and visualizations
├── configs/                # Configuration files
└── docs/                   # Documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)

### Installation
```bash
git clone https://github.com/Deepak-dv/comic-strip-generator.git
cd comic-strip-generator
pip install -r requirements.txt
```

## Evaluation Metrics

We evaluate our model using:
- **Image Quality**: FID (Fréchet Inception Distance)
- **Text Quality**: BLEU score, perplexity
- **Narrative Coherence**: Custom coherence metrics
- **Image-Text Alignment**: Cross-modal similarity scores

## Research Questions

1. Does cross-modal attention improve story coherence compared to baseline fusion methods?
2. How do GAN-based decoders compare to traditional CNNs for comic panel generation?
3. What is the impact of different sequence modeling approaches on narrative quality?

## Status

🚧 **Project in Development** - This repository is being actively developed as part of the Deep Neural Networks and Learning Systems coursework.

## Acknowledgments

This project is developed as part of the **Deep Neural Networks and Learning Systems** course assessment.

## License

This project is for educational purposes.

---

*Project by Deepak-dv | Masters in Advanced Robotics and Artificial Intelligence*
