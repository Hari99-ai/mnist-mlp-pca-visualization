# MNIST MLP Classification with PCA Visualization

This project implements a Multi-Layer Perceptron (MLP) using PyTorch to classify handwritten digits from the MNIST dataset. After training, it extracts features from the last hidden layer and applies PCA for dimensionality reduction and visualization.

## 🔧 Model Architecture

- **Input Layer**: 28x28 pixels
- **Hidden Layers**:
  - 256 units + BatchNorm + ReLU
  - 128 units + BatchNorm + ReLU
- **Output Layer**: 10 classes (digits 0–9)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam

## 🧪 Results

- **Test Accuracy**: ~97.56%  
- **Training Loss over Epochs**:

![Training Log](![image](https://github.com/user-attachments/assets/47b196c9-6090-4dc5-979e-456eb95cdb61))

- **PCA of Hidden Features**:

![PCA Plot](![image](https://github.com/user-attachments/assets/0fd18df9-917d-4de1-bde3-be8d98dafe8a))

## 📁 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/mnist-mlp-pca-visualization.git
   cd mnist-mlp-pca-visualization
