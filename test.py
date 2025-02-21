import torch
from models.clip_detector import CLIPDeepfakeDetector
from utils.dataset import DeepfakeDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, average_precision_score

# Config
BATCH_SIZE = 32

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                         (0.26862954, 0.26130258, 0.27577711))
])

# Dataset and Loader
test_dataset = DeepfakeDataset("data/celebdf/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPDeepfakeDetector().to(device)
model.load_state_dict(torch.load("clip_deepfake_detector_bs128.pth"))
model.eval()

# Inference
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze()
        preds = torch.sigmoid(outputs).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
threshold = 0.5
binary_preds = [1 if p > threshold else 0 for p in all_preds]
accuracy = accuracy_score(all_labels, binary_preds)
ap = average_precision_score(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Average Precision: {ap:.4f}")
