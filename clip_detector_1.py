import torch
import torch.nn as nn
import clip

class CLIPDeepfakeDetector(nn.Module):
    def __init__(self, clip_model="ViT-B/32", fine_tune=False):
        """
        CLIP-based deepfake detector with feature extraction capability.

        Args:
            clip_model (str): CLIP model variant (default: "ViT-B/32").
            fine_tune (bool): Whether to fine-tune the CLIP model (default: False).
        """
        super().__init__()
        self.clip_model, self.preprocess = clip.load(clip_model)
        self.classifier = nn.Linear(512, 1)  # Binary classification
        self.fine_tune = fine_tune

        # Freeze CLIP model if not fine-tuning
        if not self.fine_tune:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward_features(self, x):
        """
        Extract features from the CLIP model.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Extracted features of shape (batch_size, 512).
        """
        if self.fine_tune:
            return self.clip_model.encode_image(x).float()
        else:
            with torch.no_grad():
                return self.clip_model.encode_image(x).float()

    def forward(self, x):
        """
        Forward pass for classification.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1).
        """
        features = self.forward_features(x)
        return self.classifier(features)