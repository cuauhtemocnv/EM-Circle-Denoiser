# visualize.py
import torch
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from dataset import CircleDatasetEM
from model import Denoiser

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = CircleDatasetEM(num_samples=10)
model = Denoiser().to(device)
model.eval()  # assume trained weights loaded if you have them

noisy, clean = dataset[0]
noisy, clean = noisy.unsqueeze(0).to(device), clean.unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(noisy).cpu().squeeze().numpy()

contours = find_contours(clean.cpu().squeeze().numpy(), 0.5)

fig, axes = plt.subplots(1, 4, figsize=(16,4))
titles = ["Noisy 1", "Noisy 2", "Noisy 3", "Denoised"]
images = [noisy[0,0].cpu(), noisy[0,1].cpu(), noisy[0,2].cpu(), pred]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    for contour in contours:
        ax.plot(contour[:,1], contour[:,0], color='red', linewidth=2)

plt.tight_layout()
plt.savefig("images/example_pipeline.png", dpi=150)
plt.show()
