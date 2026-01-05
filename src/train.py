import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from datasets import load_dataset

from .model import EnhancedFramePredictor, EnhancedMultimodalLoss
from .utils import StoryDataset, get_default_transform, vocab_size


def build_dataloaders(batch_size=2, sequence_length=3,
                      max_train=800, max_val=200, max_test=100):
    dataset = load_dataset("daniel3303/StoryReasoning", streaming=True)
    transform = get_default_transform()

    train_dataset = StoryDataset(dataset["train"], sequence_length, transform, max_train)
    val_dataset = StoryDataset(dataset["train"], sequence_length, transform, max_val)
    test_dataset = StoryDataset(dataset["test"], sequence_length, transform, max_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, criterion, optimizer, device, vocab_size):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        input_images = batch["input_images"].to(device)
        target_image = batch["target_image"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        text_mask = batch["text_mask"].to(device)
        caption_tokens = batch["caption_tokens"].to(device)

        optimizer.zero_grad()

        predicted_image, caption_logits, attention_weights = model(
            input_images, text_tokens, text_mask, caption_tokens
        )

        image_loss = criterion(predicted_image, target_image, attention_weights)
        caption_loss = F.cross_entropy(
            caption_logits.reshape(-1, vocab_size),
            caption_tokens.reshape(-1),
            ignore_index=0,
        )

        loss = image_loss + 0.5 * caption_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return epoch_loss / len(dataloader)


def validate(model, dataloader, criterion, device, vocab_size):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_images = batch["input_images"].to(device)
            target_image = batch["target_image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_mask = batch["text_mask"].to(device)
            caption_tokens = batch["caption_tokens"].to(device)

            predicted_image, caption_logits, attention_weights = model(
                input_images, text_tokens, text_mask, caption_tokens
            )

            image_loss = criterion(predicted_image, target_image, attention_weights)
            caption_loss = F.cross_entropy(
                caption_logits.reshape(-1, vocab_size),
                caption_tokens.reshape(-1),
                ignore_index=0,
            )

            loss = image_loss + 0.5 * caption_loss
            val_loss += loss.item()

    return val_loss / len(dataloader)