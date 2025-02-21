import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models.clip_detector_1 import CLIPDeepfakeDetector
from utils.dataset import DeepfakeDataset

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for binary classification.

    Args:
        temperature (float): Temperature scaling parameter (default: 0.07).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Compute the supervised contrastive loss.

        Args:
            features (torch.Tensor): Extracted features of shape (batch_size, 512).
            labels (torch.Tensor): Ground truth labels of shape (batch_size).

        Returns:
            torch.Tensor: Contrastive loss.
        """
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create positive mask (same class)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()

        # Remove self-similarity
        self_mask = torch.eye(labels.size(0), device=labels.device)
        mask = mask * (1 - self_mask)

        # Compute logits
        exp_logits = torch.exp(similarity_matrix) * (1 - self_mask)  # Exclude self

        # Compute log probability
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Compute mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -mean_log_prob_pos.mean()

        return loss

# Config
BATCH_SIZE = 128
LR = 1e-5
EPOCHS = 10
TEMPERATURE = 0.07

# Transforms (CLIP preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

# Dataset and Loader
train_dataset = DeepfakeDataset("data/celebdf/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPDeepfakeDetector(fine_tune=True).to(device).float()  # Fine-tune CLIP
criterion = SupConLoss(temperature=TEMPERATURE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device).float()
        labels = labels.to(device)

        # Get features
        features = model.forward_features(images)

        # Compute loss
        loss = criterion(features, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "clip_deepfake_detector_finetuned_bs128.pth")