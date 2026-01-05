import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel


# ----------------- Vision encoder -----------------

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

        feats = self.avgpool(x)
        feats = feats.view(feats.size(0), -1)
        feats = self.fc(feats)

        return feats, [skip1, skip2, skip3, skip4]


# ----------------- U-Net decoder -----------------

class UNetDecoder(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()

        self.fc = nn.Linear(input_dim, 512 * 4 * 4)

        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.up5 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.up6 = nn.ConvTranspose2d(16, 3, 4, 2, 1)
        self.final = nn.Tanh()

    def forward(self, x, skip_connections):
        skip1, skip2, skip3, skip4 = skip_connections

        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)

        x = self.up1(x)
        skip4_r = torch.nn.functional.interpolate(skip4, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip4_r], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        skip3_r = torch.nn.functional.interpolate(skip3, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip3_r], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        skip2_r = torch.nn.functional.interpolate(skip2, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip2_r], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        skip1_r = torch.nn.functional.interpolate(skip1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip1_r], dim=1)
        x = self.conv4(x)

        x = self.up5(x)
        x = self.up6(x)
        x = self.final(x)
        return x


# ----------------- Sequence + attention -----------------

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


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

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V).squeeze(1)
        fused = visual_features + attended
        return fused, weights


# ----------------- Text + caption -----------------

class TextEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.projection(pooled)


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

        input_token = torch.full((batch_size, 1), 101, dtype=torch.long, device=fused_features.device)
        outputs = []
        hidden = (h0, c0)

        import random

        for t in range(max_len):
            embedded = self.embedding(input_token)
            lstm_out, hidden = self.lstm(embedded, hidden)
            logits = self.fc(lstm_out)
            outputs.append(logits)

            if random.random() < 0.5 and t < max_len - 1:
                input_token = target_captions[:, t + 1].unsqueeze(1)
            else:
                input_token = logits.argmax(dim=-1)

        outputs = torch.cat(outputs, dim=1)
        return outputs


# ----------------- Losses -----------------

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg)[:4])
        self.slice2 = nn.Sequential(*list(vgg)[4:9])
        self.slice3 = nn.Sequential(*list(vgg)[9:16])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        p1 = self.slice1(pred); t1 = self.slice1(target)
        p2 = self.slice2(p1);   t2 = self.slice2(t1)
        p3 = self.slice3(p2);   t3 = self.slice3(t2)

        loss = torch.mean((p1 - t1) ** 2) + \
               torch.mean((p2 - t2) ** 2) + \
               torch.mean((p3 - t3) ** 2)
        return loss


class EnhancedMultimodalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual = PerceptualLoss()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, predicted_image, target_image, attention_weights):
        l1_loss = self.l1(predicted_image, target_image)
        mse_loss = self.mse(predicted_image, target_image)
        perceptual_loss = self.perceptual(predicted_image, target_image)

        attention_entropy = -torch.mean(attention_weights * torch.log(attention_weights + 1e-8))

        total_loss = l1_loss + 0.1 * perceptual_loss + 0.05 * attention_entropy
        return total_loss


# ----------------- Full model -----------------

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
            feats, skips = self.image_encoder(input_sequence[:, t])
            image_features.append(feats)
            if t == seq_len - 1:
                decoder_skips = skips

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
