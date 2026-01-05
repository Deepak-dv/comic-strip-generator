# train.py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from google.colab import drive
from model import EnhancedFramePredictor, EnhancedMultimodalLoss  # import your model class

# -------------------------------
# Mount Google Drive
# -------------------------------
drive.mount('/content/drive', force_remount=True)
save_dir = '/content/drive/MyDrive'

# -------------------------------
# Device configuration
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Training configuration
# -------------------------------
start_epoch = 10          # Resume from this epoch
num_epochs = 20
checkpoint_path = f'{save_dir}/model_epoch_10.pth'
vocab_size = 30522  # Example vocab size, set according to your tokenizer

# -------------------------------
# Initialize model, loss, optimizer
# -------------------------------
model = EnhancedFramePredictor(vocab_size=vocab_size).to(device)
criterion = EnhancedMultimodalLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Resumed training from epoch {start_epoch}")
else:
    start_epoch = 0
    print("No checkpoint found, starting from scratch")

# -------------------------------
# Training function
# -------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, vocab_size):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc='Training')

    for batch in pbar:
        input_images = batch['input_images'].to(device)
        target_image = batch['target_image'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        text_mask = batch['text_mask'].to(device)
        caption_tokens = batch['caption_tokens'].to(device)

        optimizer.zero_grad()
        predicted_image, caption_logits, attention_weights = model(
            input_images, text_tokens, text_mask, caption_tokens
        )

        # Compute losses
        image_loss = criterion(predicted_image, target_image, attention_weights)
        caption_loss = F.cross_entropy(
            caption_logits.reshape(-1, vocab_size),
            caption_tokens.reshape(-1),
            ignore_index=0
        )

        loss = image_loss + 0.5 * caption_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return epoch_loss / len(dataloader)

# -------------------------------
# Validation function
# -------------------------------
def validate(model, dataloader, criterion, device, vocab_size):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_images = batch['input_images'].to(device)
            target_image = batch['target_image'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_mask = batch['text_mask'].to(device)
            caption_tokens = batch['caption_tokens'].to(device)

            predicted_image, caption_logits, attention_weights = model(
                input_images, text_tokens, text_mask, caption_tokens
            )

            image_loss = criterion(predicted_image, target_image, attention_weights)
            caption_loss = F.cross_entropy(
                caption_logits.reshape(-1, vocab_size),
                caption_tokens.reshape(-1),
                ignore_index=0
            )

            val_loss += (image_loss + 0.5 * caption_loss).item()

    return val_loss / len(dataloader)

# -------------------------------
# Main training loop
# -------------------------------
train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(start_epoch, num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss = train_epoch(model, train_loader, criterion, optimizer, device, vocab_size)
    val_loss = validate(model, val_loader, criterion, device, vocab_size)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{save_dir}/model_best_multimodal.pth')
        print("Saved best model!")

    # Save checkpoint every 2 epochs
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), f'{save_dir}/model_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch+1}")

# Save final model at the end
torch.save(model.state_dict(), f'{save_dir}/model_final_multimodal.pth')
print("Training complete! Final model saved.")
