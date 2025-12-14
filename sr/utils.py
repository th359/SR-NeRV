import logging
import math
import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image


class DIV2KDataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, patch_size: int = None, scale: int = 1, transform=None):
        """DIV2K HR/LR pair loader with optional patch cropping and transform."""
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))])
        assert len(self.hr_files) == len(self.lr_files), "HR and LR image counts do not match"
        self.patch_size = patch_size
        self.scale = scale
        self.transform = transform
        if self.transform is None:
            import torchvision.transforms as T
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def lr_argument(self, lr_img, r=3):
        """Apply light augmentation on the low-resolution image before training."""
        # Smooth the LR image to mimic downsampling artifacts
        # lr_img = lr_img.filter(ImageFilter.GaussianBlur(radius=r)) # SR-NeRV
        lr_img = lr_img.filter(ImageFilter.BLUR) # SR-HNeRV
        if random.random() < 0.3:
            w, h = lr_img.size
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = min(w, x1 + random.randint(20, w // 4))
            y2 = min(h, y1 + random.randint(20, h // 4))
            region = lr_img.crop((x1, y1, x2, y2))
            enhancer = ImageEnhance.Color(region)
            factor = random.uniform(0.5, 1.5)
            region = enhancer.enhance(factor)
            lr_img.paste(region, (x1, y1))
        return lr_img

    def __getitem__(self, idx: int):
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        if self.patch_size:
            hr_width, hr_height = hr_img.width, hr_img.height
            lr_width, lr_height = lr_img.width, lr_img.height
            lr_patch_size = self.patch_size // self.scale
            x_lr = random.randint(0, lr_width - lr_patch_size)
            y_lr = random.randint(0, lr_height - lr_patch_size)
            x_hr = x_lr * self.scale
            y_hr = y_lr * self.scale
            # Crop corresponding patches from LR and HR images
            hr_img = hr_img.crop((x_hr, y_hr, x_hr + self.patch_size, y_hr + self.patch_size))
            lr_img = lr_img.crop((x_lr, y_lr, x_lr + lr_patch_size, y_lr + lr_patch_size))
        lr_img = self.lr_argument(lr_img)
        hr_tensor = self.transform(hr_img)  # shape: (3, H, W), range [0.0, 1.0]
        lr_tensor = self.transform(lr_img)
        return lr_tensor, hr_tensor


def create_unique_directory(base_dir):
    """Create a unique directory by appending an incrementing suffix if needed."""
    target_dir = base_dir
    counter = 1
    while os.path.exists(target_dir):
        target_dir = f"{base_dir}_{counter}"
        counter += 1
    os.makedirs(target_dir)
    return target_dir


def get_warmup_multistep_scheduler(optimizer, warmup_epochs, milestones, gamma, base_lr):
    """Warmup scheduler followed by MultiStep-like decay."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linearly increasing (1/warmup_epochs -> 1)
        else:
            # MultiStepLR-like decay once warmup has finished
            steps = sum(epoch >= m for m in milestones)
            return gamma ** steps
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def calc_psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate PSNR (dB) between two images.
    Expects tensors of shape (C, H, W) within [0, 1].
    """
    sr_np = sr.detach().cpu().numpy()
    hr_np = hr.detach().cpu().numpy()
    mse = np.mean((sr_np - hr_np) ** 2)
    if mse == 0:
        return float('inf')
    # PSNR = 10 * log10(max_val^2 / MSE)
    return 10 * math.log10((max_val ** 2) / mse)


def calc_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Compute SSIM between two images.
    Expects tensors of shape (C, H, W) normalized to [0, 1].
    """
    sr_np = sr.detach().cpu().numpy()
    hr_np = hr.detach().cpu().numpy()
    # Rearrange axes from (C,H,W) -> (H,W,C)
    sr_np = sr_np.transpose(1, 2, 0)
    hr_np = hr_np.transpose(1, 2, 0)
    h, w, _ = sr_np.shape
    win_size = 7
    if min(h, w) < win_size:
        win_size = min(h, w) // 2 * 2 + 1  # Ensure an odd window size (e.g., 6 -> 5)
    # Use structural_similarity for multi-channel SSIM
    ssim_val = structural_similarity(hr_np, sr_np, multichannel=True, data_range=1.0, win_size=win_size,)
    return float(ssim_val)


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, epoch: int, device: str = 'cuda', output: str = 'output'):
    """
    Evaluate model on the validation dataloader and return mean PSNR/SSIM.
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    flag = True
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            sr_imgs = torch.clamp(sr_imgs, 0.0, 1.0)
            for i in range(sr_imgs.size(0)):
                psnr_val = calc_psnr(sr_imgs[i], hr_imgs[i])
                # ssim_val = calc_ssim(sr_imgs[i], hr_imgs[i])
                total_psnr += psnr_val
                # total_ssim += ssim_val
                count += 1
            if flag:
                lr = lr_imgs[i].cpu()
                sr = sr_imgs[i].cpu()
                hr = hr_imgs[i].cpu()
                _, h, w = hr.shape
                lr_resized = torch.nn.functional.interpolate(lr.unsqueeze(0), size=(h, w), mode='bicubic', align_corners=False).squeeze(0)
                combined = torch.cat([lr_resized, sr, hr], dim=2)
                save_path = os.path.join(output, 'eval_img')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_img_path = os.path.join(save_path, f"epoch_{epoch}.png")
                # Save the first visual comparison for quick sanity checks
                to_pil_image(combined).save(save_img_path)
                flag = False
    model.train()
    return total_psnr / count, total_ssim / count


def setup_logger(log_file: str = "train.log"):
    """Configure logging to stream INFO+ messages to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w'),
        ],
    )
