from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

CLASSES = ["bullish", "bearish", "neutral"]
NUM_CLASSES = len(CLASSES)


class TinyCNN(nn.Module):
    """A tiny CNN – replace with your preferred model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*56*56, 64)
        self.fc2 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 112x112
        x = self.pool(F.relu(self.conv2(x)))  # 56x56
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model(weights_path: Path | None) -> Tuple[nn.Module, transforms.Compose]:
    model = TinyCNN()
    model.eval()
    if weights_path and weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return model, preprocess


def predict_pil(model: nn.Module, preprocess, img: Image.Image) -> Dict:
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        idx = int(torch.tensor(probs).argmax().item())
        return {
            "label": CLASSES[idx],
            "probs": {c: float(probs[i]) for i,c in enumerate(CLASSES)}
        }