import re
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size


class StoryDataset(Dataset):
    def __init__(self, hf_dataset, sequence_length=3, transform=None, max_stories=800):
        self.sequence_length = sequence_length
        self.transform = transform
        self.tokenizer = tokenizer
        self.sequences = []

        story_count = 0
        for story in hf_dataset:
            if story_count >= max_stories:
                break

            num_frames = len(story["images"])
            if num_frames >= sequence_length + 1:
                for i in range(num_frames - sequence_length):
                    self.sequences.append(
                        {"story_data": story, "start_idx": i}
                    )

            story_count += 1

        print(f"Created {len(self.sequences)} sequences from {story_count} stories")

    def extract_text(self, grounded_story):
        if not grounded_story:
            return ""
        clean = re.sub(r"<[^>]+>", "", grounded_story)
        return clean.strip()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        story = seq_info["story_data"]
        start_idx = seq_info["start_idx"]

        input_frames = []
        for i in range(self.sequence_length):
            img = story["images"][start_idx + i]
            if self.transform:
                img = self.transform(img)
            input_frames.append(img)

        target_img = story["images"][start_idx + self.sequence_length]
        if self.transform:
            target_img = self.transform(target_img)

        narrative = self.extract_text(story.get("story", ""))

        text_encoding = self.tokenizer(
            narrative[:512],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        caption_text = narrative[:200] if narrative else "A comic panel"
        caption_encoding = self.tokenizer(
            caption_text,
            max_length=50,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_images": torch.stack(input_frames),
            "target_image": target_img,
            "text_tokens": text_encoding["input_ids"].squeeze(0),
            "text_mask": text_encoding["attention_mask"].squeeze(0),
            "caption_tokens": caption_encoding["input_ids"].squeeze(0),
            "caption_mask": caption_encoding["attention_mask"].squeeze(0),
        }


def get_default_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def enhance(img):
    img = (img + 1) / 2
    img = img * 2.5
    img = torch.clamp(img, 0, 1)
    img = (img * 2) - 1
    return img


def denorm(img):
    return torch.clamp((img + 1) / 2, 0, 1)
