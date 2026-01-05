# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

# -------------------------------
# ResNet Encoder for Image Features
# -------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        # skip connections for U-Net decoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip1 = x

        x = self.maxpool(x)
        x = self.layer1(x)
        skip2 = x

        x = self.layer2(x)
        skip3 = x

        x = self.layer3(x)
        skip4 = x

        x = self.layer4(x)

        # global features
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)

        return features, [skip1, skip2, skip3, skip4]

# -------------------------------
# U-Net Decoder
# -------------------------------
class UNetDecoder(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.fc = nn.Linear(input_dim, 512 * 4 * 4)

        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.up5 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.up6 = nn.ConvTranspose2d(16, 3, 4, 2, 1)
        self.final = nn.Tanh()

    def forward(self, x, skips):
        skip1, skip2, skip3, skip4 = skips
        x = self.fc(x).view(x.size(0), 512, 4, 4)

        x = self.up1(x)
        skip4_resized = nn.functional.interpolate(skip4, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv1(torch.cat([x, skip4_resized], dim=1))

        x = self.up2(x)
        skip3_resized = nn.functional.interpolate(skip3, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv2(torch.cat([x, skip3_resized], dim=1))

        x = self.up3(x)
        skip2_resized = nn.functional.interpolate(skip2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv3(torch.cat([x, skip2_resized], dim=1))

        x = self.up4(x)
        skip1_resized = nn.functional.interpolate(skip1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv4(torch.cat([x, skip1_resized], dim=1))

        x = self.up5(x)
        x = self.up6(x)
        x = self.final(x)
        return x

# -------------------------------
# LSTM Sequence Encoder
# -------------------------------
class SequenceEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]

# -------------------------------
# Cross-Modal Attention (Visual + Text + Sequence)
# -------------------------------
class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.query_visual = nn.Linear(feature_dim, feature_dim)
        self.key_visual = nn.Linear(feature_dim, feature_dim)
        self.value_visual = nn.Linear(feature_dim, feature_dim)
        self.key_text = nn.Linear(feature_dim, feature_dim)
        self.value_text = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5

    def forward(self, visual_features, text_features, sequence_features):
        Q = self.query_visual(visual_features).unsqueeze(1)
        K_seq = self.key_visual(sequence_features).unsqueeze(1)
        V_seq = self.value_visual(sequence_features).unsqueeze(1)
        K_txt = self.key_text(text_features).unsqueeze(1)
        V_txt = self.value_text(text_features).unsqueeze(1)

        K = torch.cat([K_seq, K_txt], dim=1)
        V = torch.cat([V_seq, V_txt], dim=1)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, V).squeeze(1)

        fused_features = visual_features + attended_features
        return fused_features, attention_weights

# -------------------------------
# Text Encoder using BERT
# -------------------------------
class TextEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.projection(pooled)

# -------------------------------
# Caption Decoder with LSTM
# -------------------------------
class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, fused_features, target_captions):
        batch_size = fused_features.size(0)
        max_len = target_captions.size(1)
        h0 = fused_features.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        input_token = torch.full((batch_size, 1), 101, dtype=torch.long, device=fused_features.device)
        outputs = []

        for t in range(max_len):
            embedded = self.embedding(input_token)
            lstm_out, hidden = self.lstm(embedded, hidden)
            logits = self.fc(lstm_out)
            outputs.append(logits)
            if torch.rand(1).item() < 0.5 and t < max_len - 1:
                input_token = target_captions[:, t+1].unsqueeze(1)
            else:
                input_token = logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)

# -------------------------------
# Full Frame Predictor Model
# -------------------------------
class EnhancedFramePredictor(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.image_encoder = ResNetEncoder(output_dim=512)
        self.text_encoder = TextEncoder(output_dim=512)
        self.sequence_encoder = SequenceEncoder(input_dim=512, hidden_dim=512)
        self.attention = CrossModalAttention(feature_dim=512)
        self.decoder = UNetDecoder(input_dim=512)
        self.caption_decoder = CaptionDecoder(vocab_size=vocab_size)

    def forward(self, input_sequence, text_tokens, text_mask, caption_tokens=None):
        batch_size, seq_len = input_sequence.size(0), input_sequence.size(1)
        image_features = []

        for t in range(seq_len):
            features, skip_connections = self.image_encoder(input_sequence[:, t])
            image_features.append(features)
            if t == seq_len - 1:
                decoder_skips = skip_connections

        image_features = torch.stack(image_features, dim=1)
        text_features = self.text_encoder(text_tokens, text_mask)
        sequence_features = self.sequence_encoder(image_features)

        last_image_features = image_features[:, -1]
        fused_features, attention_weights = self.attention(
            last_image_features, text_features, sequence_features
        )

        predicted_image = self.decoder(fused_features, decoder_skips)

        if caption_tokens is not None:
            caption_logits = self.caption_decoder(fused_features, caption_tokens)
            return predicted_image, caption_logits, attention_weights

        return predicted_image, None, attention_weights
