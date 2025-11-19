# Comic Strip Generator

A multimodal deep learning project for my Deep Neural Networks and Learning Systems course. I'm building a system that can look at a sequence of comic panels and predict what comes next - both the image and the dialog!

## What I'm Building

Basically, I'm training a neural network to understand comic stories. You give it a few panels from a comic strip, and it generates the next panel with both:
- The image (what the characters and scene look like)
- The text (what they're saying or what's happening)

The cool part is that the model needs to understand both the visual story and the text together to make sensible predictions.

## Why This Project?

I chose this because:
- Comics are a unique challenge - they mix visual art with narrative text
- Most AI models struggle to keep both modalities coherent
- It's more interesting than just doing image or text generation separately
- Could actually be useful for assistive tech or helping artists brainstorm ideas

## My Approach

I'm implementing a few key components:

**For Processing Input:**
- **Visual Encoder**: Using a CNN (probably ResNet) to extract features from the comic panel images
- **Text Encoder**: LSTM or Transformer to process the dialog and descriptions
- **Cross-Modal Attention**: This is my main innovation - it lets the image features and text features "talk" to each other so the model understands connections (like when a character mentions coffee and there's a coffee cup in the image)

**For Understanding Sequence:**
- **Sequence Model**: LSTM/GRU to understand the temporal flow of the story across multiple panels

**For Generating Output:**
- **Image Decoder**: Using a GAN to generate the next comic panel image
- **Text Decoder**: Autoregressive decoder to generate the dialog

## Dataset

I'm using the StoryReasoning Dataset which has:
- Sequential image-text pairs designed for story understanding
- Annotations for objects, locations, actions, and descriptions
- Different narrative styles

## Technical Setup

I'm developing this in **Google Colab** because:
- Free GPU access (T4/P100 depending on availability)
- Don't need to worry about local setup
- Easy to share notebooks with my prof if needed

### How to Run

If you want to check out my code:

```python
# Clone this repo in Colab
!git clone https://github.com/Deepak-dv/comic-strip-generator.git
%cd comic-strip-generator

# Install dependencies
!pip install -r requirements.txt

# Make sure you enable GPU in Colab
# Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

## Research Questions I'm Exploring

1. Does cross-modal attention actually improve story coherence compared to just concatenating image and text features?
2. How well do GANs work for generating comic-style images vs traditional decoders?
3. Can the model learn different comic styles or does it just average everything?

## Evaluation

I'm planning to measure:
- **Image Quality**: FID score to see if generated images look realistic
- **Text Quality**: BLEU score and perplexity for the generated dialog
- **Coherence**: Custom metric to check if the next panel actually makes sense given the story
- **Alignment**: How well the generated image matches the generated text

## Current Status

🚧 **Just started** - Currently setting up the project structure and reviewing the dataset. Will update as I make progress.

## Project Structure

```
comic-strip-generator/
├── data/                    # Dataset
├── notebooks/              # Colab notebooks for experiments
├── models/                 # Model architecture code
├── results/                # Generated samples and metrics
└── configs/                # Training configs
```

## Course Info

This is my final project for the Deep Neural Networks and Learning Systems course (Masters in Artificial Intelligence).

---

*Feel free to reach out if you have questions or suggestions!*
