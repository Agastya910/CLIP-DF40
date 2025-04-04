# Deepfake Detection Using CLIP

A deepfake detection system built using OpenAI's CLIP model as a feature extractor, fine-tuned with a custom contrastive loss for better feature separation. The extracted features are used to train SVM classifiers that can accurately detect manipulated videos. This project was evaluated on the **Celeb-DF** dataset.

## 📌 Highlights

- 🔍 **Feature Extraction with CLIP**: Leveraged pretrained CLIP model for extracting robust multi-modal embeddings from video frames.
- 🎯 **Contrastive Fine-Tuning**: Applied a custom contrastive loss to improve separation between real and fake embeddings in the feature space.
- 🧠 **SVM Classifier**: Trained a Support Vector Machine on the learned features to classify videos as real or fake.
- 📈 **High Accuracy**: Achieved strong performance metrics (accuracy, precision, recall) on the Celeb-DF benchmark dataset.

## 🛠️ Tech Stack

- Python
- PyTorch
- OpenAI CLIP
- scikit-learn (SVM)
- Celeb-DF dataset

## 📁 Project Structure

```
deepfake-detection-clip/
├── data/                     # Processed frames or embeddings
├── models/                   # Saved CLIP and SVM models
├── src/
│   ├── extract_features.py   # Extract features using CLIP
│   ├── contrastive_loss.py   # Custom contrastive loss function
│   ├── train_clip.py         # Fine-tune CLIP with contrastive loss
│   ├── train_svm.py          # Train SVM on CLIP embeddings
│   └── evaluate.py           # Evaluate performance on test data
├── notebooks/                # Jupyter notebooks for experiments
├── requirements.txt
└── README.md
```

## 📊 Dataset

- **Celeb-DF (v2)**: A large-scale deepfake dataset containing high-quality deepfake videos of celebrities.
- Preprocessing involved extracting representative frames from each video and resizing them for input into the CLIP model.

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/deepfake-detection-clip.git
cd deepfake-detection-clip
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download and prepare the Celeb-DF dataset

Follow instructions from the [official Celeb-DF website](https://github.com/yuezunli/Celeb-DF) to request access and download the dataset.

### 4. Extract features using CLIP
```bash
python src/extract_features.py --data_path data/frames --output_path data/features
```

### 5. Fine-tune CLIP (optional)
```bash
python src/train_clip.py --data_path data/features --loss contrastive
```

### 6. Train and evaluate SVM
```bash
python src/train_svm.py --features_path data/features --output_path models/svm.pkl
python src/evaluate.py --model_path models/svm.pkl --test_data data/test_features
```

## 📈 Results

| Metric      | Score     |
|-------------|-----------|
| Accuracy    | 93.5%     |
| Precision   | 94.2%     |
| Recall      | 92.8%     |
| F1 Score    | 93.5%     |

*Evaluated on the Celeb-DF v2 test set.*

## 📚 References

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Celeb-DF: A Large-scale Dataset for DeepFake Forensics](https://arxiv.org/abs/2003.07590)


```
