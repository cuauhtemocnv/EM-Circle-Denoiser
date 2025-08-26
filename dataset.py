# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.draw import disk, polygon
from scipy.ndimage import shift, gaussian_filter
import random

class CircleDatasetEM(Dataset):
    def __init__(self, num_samples=500, size=128, noise_level=0.05,
                 max_shift=2, add_triangles=True):
        self.num_samples = num_samples
        self.size = size
        self.noise_level = noise_level
        self.max_shift = max_shift
        self.add_triangles = add_triangles

    def __len__(self):
        return self.num_samples

    def make_circle(self):
        img = np.zeros((self.size, self.size), dtype=np.float32)
        r = np.random.randint(self.size // 6, self.size // 4)
        center = (np.random.randint(r, self.size - r), np.random.randint(r, self.size - r))
        rr, cc = disk(center, r, shape=img.shape)
        img[rr, cc] = 1.0
        return img

    def add_triangle(self, img):
        pts = np.array([
            [np.random.randint(0, self.size), np.random.randint(0, self.size)],
            [np.random.randint(0, self.size), np.random.randint(0, self.size)],
            [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        ])
        rr, cc = polygon(pts[:, 0], pts[:, 1], shape=img.shape)
        img[rr, cc] = 1.0
        return img

    def apply_defocus(self, img, sigma=None):
        if sigma is None:
            sigma = np.random.uniform(0.5, 1.5)
        return gaussian_filter(img, sigma=sigma)

    def apply_chromatic_aberration(self, img):
        sigma_x = np.random.uniform(0.8, 1.2)
        sigma_y = np.random.uniform(0.8, 1.2)
        return gaussian_filter(img, sigma=(sigma_x, sigma_y))

    def apply_mtf_noise(self, img):
        mtf_sigma = 0.3
        snr = self.noise_level
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        radius = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        mtf = np.exp(-(radius / (rows * mtf_sigma)) ** 2)
        fshift_filtered = fshift * mtf
        img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered)).real
        img_filtered = np.clip(img_filtered, 0, 1)
        img_noisy = img_filtered + np.random.normal(0, snr, img.shape)
        return np.clip(img_noisy, 0, 1)

    def apply_shift(self, img):
        dy, dx = np.random.randint(-self.max_shift, self.max_shift + 1, size=2)
        return shift(img, shift=(dy, dx), order=0, mode="constant", cval=0.0)

    def add_em_effects(self, img):
        img = self.apply_shift(img)
        img = self.apply_defocus(img)
        img = self.apply_chromatic_aberration(img)
        img = self.apply_mtf_noise(img)
        return img

    def __getitem__(self, idx):
        clean = self.make_circle()
        noisy_imgs = []
        for _ in range(3):
            img = clean.copy()
            if self.add_triangles and random.random() < 0.7:
                img = self.add_triangle(img)
            img = self.add_em_effects(img)
            noisy_imgs.append(img)
        noisy_imgs = np.stack(noisy_imgs, axis=0)
        return torch.tensor(noisy_imgs, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
