# train.py
import torch
from torch.utils.data import DataLoader
from dataset import CircleDatasetEM
from model import Denoiser
from utils import weighted_mse

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CircleDatasetEM(num_samples=2000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Denoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            loss = weighted_mse(out, clean, weight=10.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss = {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    train_model()
