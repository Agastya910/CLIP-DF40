import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.clip_detector import CLIPDeepfakeDetector
from utils.dataset import DeepfakeDataset
from torchvision import transforms

# Config
BATCH_SIZE = 128
LR = 1e-5
EPOCHS = 10

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
model = CLIPDeepfakeDetector().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images = images.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "clip_deepfake_detector_bs128.pth")
