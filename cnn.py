import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
])


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transform),
    batch_size=1000, shuffle=False
)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.fc3(x)  # logits
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, loader):
    model.train()
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}: Loss = {loss.item():.4f}')

train(model, train_loader)

def test(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    print(f'Test Accuracy: {correct / len(loader.dataset):.4f}')

test(model, test_loader)


# Get features from the last hidden layer
def extract_features(model, loader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            x = data.view(-1, 28*28)
            x = F.relu(model.fc1(x))
            x = F.relu(model.fc2(x))
            features.append(x.cpu())
            labels.append(target)

    return torch.cat(features), torch.cat(labels)

features, labels = extract_features(model, test_loader)

# Apply PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(features.numpy())

# Plot
plt.figure(figsize=(8,6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels.numpy(), cmap='tab10', alpha=0.6)
plt.title('PCA of MNIST MLP Features')
plt.colorbar()
plt.show()
