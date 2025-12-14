import argparse
import logging
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

import utils
from model import ResidualSRModel


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight SR model on DIV2K", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument('--data_dir',       type=str,   default='DIV2K',    help='Root directory containing DIV2K train/val folders')
    parser.add_argument('--scale',          type=int,   default=2,          help='Upscaling factor (expects DIV2K bicubic LR folder to match)')
    parser.add_argument('--patch_size',     type=int,   default=96,         help='Training crop size on HR images')
    parser.add_argument('--batch_size',     type=int,   default=16,         help='Mini-batch size for training')
    parser.add_argument('--epochs',         type=int,   default=100,        help='Number of training epochs')
    parser.add_argument('--eval_interval',  type=int,   default=25,         help='Validate and checkpoint every N epochs')
    parser.add_argument('--device',         type=str,   default='cuda',     choices=['cuda', 'cpu'], help='Compute device')
    parser.add_argument('--output',         type=str,   default='output',   help='Output directory for logs, images, and checkpoints')
    parser.add_argument('--lr',             type=float, default=1e-4,       help='Base learning rate')
    parser.add_argument('--warmup_epochs',  type=int,   default=5,          help='Warmup epochs before stepping the LR schedule')
    parser.add_argument('--lr_milestones',  type=int,   nargs='+',          default=[200, 400, 600], help='Epoch milestones for LR decay')
    parser.add_argument('--lr_gamma',       type=float, default=0.5,        help='Multiplicative LR decay at each milestone')
    args = parser.parse_args()
    train(args)


def train(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("use cpu")  # Fallback when CUDA is unavailable
        args.device = 'cpu'

    utils.create_unique_directory(args.output)
    log_path = os.path.join(args.output, "training.log")
    utils.setup_logger(log_file=log_path)
    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Device: {args.device}")

    train_hr_dir = os.path.join(args.data_dir, f"DIV2K_train_HR")
    train_lr_dir = os.path.join(args.data_dir, f"DIV2K_train_LR_bicubic/X{args.scale}")
    val_hr_dir = os.path.join(args.data_dir, f"DIV2K_valid_HR")
    val_lr_dir = os.path.join(args.data_dir, f"DIV2K_valid_LR_bicubic/X{args.scale}")

    train_dataset = utils.DIV2KDataset(train_hr_dir, train_lr_dir, patch_size=args.patch_size, scale=args.scale)
    val_dataset = utils.DIV2KDataset(val_hr_dir, val_lr_dir, patch_size=None, scale=args.scale)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = ResidualSRModel(scale_factor=args.scale)
    logging.info(model)
    model_param = sum(p.data.nelement() for p in model.parameters()) / 1e6
    logging.info(f"ModelParam: {model_param}M")
    model.to(args.device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = utils.get_warmup_multistep_scheduler(
        optimizer, warmup_epochs=args.warmup_epochs, milestones=args.lr_milestones, gamma=args.lr_gamma, base_lr=args.lr,
    )

    best_psnr = 0.0
    for epoch in range(args.epochs):
        model.train()
        count = 0
        total_psnr = 0.0
        total_loss = 0.0
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(args.device)
            hr_imgs = hr_imgs.to(args.device)
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for i in range(sr_imgs.size(0)):
                psnr_val = utils.calc_psnr(sr_imgs[i], hr_imgs[i])
                total_psnr += psnr_val
                count += 1
        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / count
        scheduler.step()
        logging.info(f"Epoch {epoch+1}/{args.epochs} - PSNR: {avg_psnr:.2f} - Avg Train Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            lr = lr_imgs[0].cpu()
            sr = sr_imgs[0].cpu()
            hr = hr_imgs[0].cpu()
            _, h, w = hr.shape
            lr_resized = torch.nn.functional.interpolate(lr.unsqueeze(0), size=(h, w), mode='bicubic', align_corners=False).squeeze(0)
            combined = torch.cat([lr_resized, sr, hr], dim=2)
            save_path = os.path.join(args.output, 'train_img')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_img_path = os.path.join(save_path, f"epoch_{epoch}.png")
            # Snapshot a training triplet for quick visual inspection
            to_pil_image(combined).save(save_img_path)

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            psnr, ssim = utils.evaluate(model, val_loader, epoch=epoch, device=args.device, output=args.output)
            logging.info(f"Validation - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            if psnr > best_psnr:
                best_psnr = psnr
                save_model_path = os.path.join(args.output, "best_model.pth")
                torch.save(model.state_dict(), save_model_path)
                logging.info(f"  * New best model saved (epoch {epoch+1}) with PSNR {psnr:.2f} dB")
            save_model_path = os.path.join(args.output, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_model_path)

    save_model_path = os.path.join(args.output, "last_model.pth")
    torch.save(model.state_dict(), save_model_path)
    logging.info("Training complete! Model saved to last_model.pth")


if __name__ == "__main__":
    main()
