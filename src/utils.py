# utils.py
import torch
import textwrap
from transformers import BertTokenizer

# -------------------------------
# Image normalization / denormalization
# -------------------------------
def denorm(img: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from [-1,1] to [0,1] range for visualization.
    """
    return torch.clamp((img + 1) / 2, 0, 1)


def enhance(img: torch.Tensor) -> torch.Tensor:
    """
    Apply simple enhancement to predicted images.
    Scales the image to emphasize contrast and clamps to [-1, 1].
    """
    img = (img + 1) / 2
    img = img * 2.5
    img = torch.clamp(img, 0, 1)
    img = (img * 2) - 1
    return img


# -------------------------------
# Text decoding
# -------------------------------
def decode_text(token_ids, tokenizer_name='bert-base-uncased', max_sentences=4):
    """
    Decode token IDs into a list of sentences (for captions).
    Pads or truncates to max_sentences.
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    token_ids = token_ids.cpu().numpy()
    full_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    # Split into sentences
    sentences = [s.strip() + '.' for s in full_text.split('.') if s.strip()]

    # Pad or truncate to max_sentences
    if len(sentences) >= max_sentences:
        caps = sentences[:max_sentences]
    else:
        caps = sentences + [f"Frame {i+1}" for i in range(len(sentences), max_sentences)]

    return caps


# -------------------------------
# Utility for wrapping text for plotting
# -------------------------------
def wrap_caption(text, width=40):
    """
    Wrap caption text to fit inside a plot title or axis.
    """
    return "\n".join(textwrap.wrap(text, width))
