import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
with open("data/processed/le.pkl", "rb") as f:
    le = pickle.load(f)
class_weights = torch.tensor(np.load("data/processed/class_weights.npy"), dtype=torch.float)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

device = torch.device("cpu")

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X.shape[1]
num_classes = len(le.classes_)
model = EmotionClassifier(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 140
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    preds = torch.argmax(val_outputs, dim=1)
    acc = (preds == y_val).float().mean().item()
print(f"Validation Accuracy: {acc:.4f}")

torch.save(model.state_dict(), "models/emotion_classifier.pt")
print("Model saved to models/emotion_classifier.pt")
