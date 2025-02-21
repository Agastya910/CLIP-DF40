import clip
import torch.nn as nn
import torch

class CLIPDeepfakeDetector(nn.Module):
    def __init__(self, clip_model="ViT-B/32"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(clip_model)
        self.classifier = nn.Linear(512, 1)  # Ensure classifier is in float32
        
    def forward(self, x):
        with torch.no_grad():
            # Cast CLIP output to float32
            image_features = self.clip_model.encode_image(x).float()
        return self.classifier(image_features)