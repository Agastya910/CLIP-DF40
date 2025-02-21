import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, average_precision_score

from models.clip_detector_1 import CLIPDeepfakeDetector  # Updated import
from utils.dataset import DeepfakeDataset

# Config
BATCH_SIZE = 32

# Define the transforms (must match those used in fine-tuning)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                         (0.26862954, 0.26130258, 0.27577711))
])

# Load the training dataset for feature extraction
train_dataset = DeepfakeDataset("data/celebdf/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the CLIP model and load the fine-tuned weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPDeepfakeDetector(clip_model="ViT-B/32").to(device)
model.load_state_dict(torch.load("clip_deepfake_detector_finetuned_bs128.pth", map_location=device))
model.eval()

# Extract features for the training set
train_features = []
train_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        features = model.forward_features(images)  # Use forward_features
        features = features.view(features.size(0), -1).cpu().numpy()
        train_features.append(features)
        train_labels.append(labels.cpu().numpy())

X_train = np.concatenate(train_features, axis=0)
y_train = np.concatenate(train_labels, axis=0)

# Train the linear SVM classifier
svm_classifier = LinearSVC(max_iter=10000)
svm_classifier.fit(X_train, y_train)

# Load the test dataset for evaluation
test_dataset = DeepfakeDataset("data/celebdf/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_features = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        features = model.forward_features(images)  # Use forward_features
        features = features.view(features.size(0), -1).cpu().numpy()
        test_features.append(features)
        test_labels.append(labels.cpu().numpy())

X_test = np.concatenate(test_features, axis=0)
y_test = np.concatenate(test_labels, axis=0)

# Obtain decision scores from the SVM
svm_scores = svm_classifier.decision_function(X_test)
# Apply a threshold of 0: positive scores -> class 1, negative -> class 0
binary_preds = [1 if score > 0 else 0 for score in svm_scores]

# Calculate metrics
accuracy = accuracy_score(y_test, binary_preds)
ap = average_precision_score(y_test, svm_scores)

print(f"Accuracy: {accuracy:.4f}")
print(f"Average Precision: {ap:.4f}")