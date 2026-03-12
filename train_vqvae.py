import os
import uuid
from datetime import datetime
import cv2
import glob
import json
import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from transformers import SegformerForSemanticSegmentation

from tqdm import tqdm
from models import VQVAE

MASK_COORDS_PATH = "mask_coords.json"
PRIORITY_COORDS_PATH = "priority_coords.json"

# Cityscapes 19-class color palette (RGB)
CITYSCAPES_COLORS = torch.tensor([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=torch.float32) / 255.0  # (19, 3)

DEVICE = "cuda"
DATA_PATH = "dataset_raw/images/*.jpg"
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PRIORITY_WEIGHT = 100.0  # How much more to weight the priority region
SEG_LOSS_WEIGHT = 0.5
LPIPS_LOSS_WEIGHT = 0.1
SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

# ImageNet normalization for SegFormer input
SEG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
SEG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DriveDataset(Dataset):
    def __init__(self, glob_pattern, mask_coords_path=MASK_COORDS_PATH):
        super().__init__()
        self.files = sorted(glob.glob(glob_pattern))
        print(f"Found {len(self.files)} images.")

        self.mask_coords = None
        if os.path.exists(mask_coords_path):
            with open(mask_coords_path, "r") as f:
                self.mask_coords = json.load(f)
            print(f"Mask enabled: {self.mask_coords}")
        else:
            print("No mask_coords.json found - using full images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        if img is None:
            raise ValueError(f"Failed to load image: {self.files[idx]}")

        if self.mask_coords is not None:
            x1 = self.mask_coords["x1"]
            y1 = self.mask_coords["y1"]
            x2 = self.mask_coords["x2"]
            y2 = self.mask_coords["y2"]
            img[y1:y2, x1:x2] = 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = torch.FloatTensor(img).permute(2, 0, 1)

        return img


def colorize_labels(labels):
    """Convert (B, H, W) int label tensor to (B, 3, H, W) RGB using cityscapes palette."""
    palette = CITYSCAPES_COLORS.to(labels.device)  # (19, 3)
    labels_clamped = labels.clamp(0, len(palette) - 1)
    colored = palette[labels_clamped]  # (B, H, W, 3)
    return colored.permute(0, 3, 1, 2)  # (B, 3, H, W)


def save_seg_maps(segformer, sample, recon, seg_mean, seg_std, path):
    """Save a side-by-side grid of original and reconstructed segmentation maps."""
    orig_labels = get_pseudo_labels(segformer, sample, seg_mean, seg_std)
    recon_labels = get_pseudo_labels(segformer, recon.clamp(0, 1), seg_mean, seg_std)
    orig_colored = colorize_labels(orig_labels)
    recon_colored = colorize_labels(recon_labels)
    comparison = torch.cat([orig_colored, recon_colored], dim=0)
    save_image(comparison, path, nrow=len(sample))


def get_pseudo_labels(segformer, batch, seg_mean, seg_std):
    normalized = (batch - seg_mean) / seg_std
    with torch.no_grad():
        logits = segformer(pixel_values=normalized).logits  # (B, 19, H/4, W/4)
    return F.interpolate(logits, size=batch.shape[-2:], mode='bilinear', align_corners=False).argmax(dim=1)


def loss_fn(recon_x, x, vq_loss, seg_logits, pseudo_labels, lpips_loss, weight_mask=None):
    if weight_mask is not None:
        mse_per_pixel = (recon_x - x) ** 2
        recon_loss = (mse_per_pixel * weight_mask).mean()
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    seg_loss = F.cross_entropy(seg_logits, pseudo_labels)
    total_loss = recon_loss + vq_loss * x.size(0) + SEG_LOSS_WEIGHT * seg_loss + LPIPS_LOSS_WEIGHT * lpips_loss
    return total_loss, recon_loss, seg_loss


def create_priority_mask(height, width, priority_coords, priority_weight, device):
    """Create a weight mask with higher values in the priority region."""
    mask = torch.ones(1, 1, height, width, device=device)
    if priority_coords is not None:
        x1, y1 = priority_coords["x1"], priority_coords["y1"]
        x2, y2 = priority_coords["x2"], priority_coords["y2"]
        mask[:, :, y1:y2, x1:x2] = priority_weight
    return mask


def main():
    print(f"Training on {DEVICE}...")

    # Load priority coords if available
    priority_coords = None
    if os.path.exists(PRIORITY_COORDS_PATH):
        with open(PRIORITY_COORDS_PATH, "r") as f:
            priority_coords = json.load(f)
        print(f"Priority region enabled: {priority_coords} (weight={PRIORITY_WEIGHT})")
    else:
        print("No priority_coords.json found - using uniform loss weighting")

    # Load frozen LPIPS perceptual loss (VGG)
    lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # Load frozen SegFormer for pseudo-label generation
    print(f"Loading SegFormer ({SEGFORMER_MODEL})...")
    segformer = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    segformer = segformer.to(DEVICE)  # type: ignore[arg-type]
    segformer.eval()
    for p in segformer.parameters():
        p.requires_grad_(False)
    seg_mean = SEG_MEAN.to(DEVICE)
    seg_std = SEG_STD.to(DEVICE)

    # Setup
    dataset = DriveDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create priority mask (assumes all images are same size)
    sample_img = dataset[0]
    _, height, width = sample_img.shape
    weight_mask = create_priority_mask(
        height, width, priority_coords, PRIORITY_WEIGHT, DEVICE
    )

    # Create unique run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    run_dir = os.path.join("runs", run_id)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    samples_dir = os.path.join(run_dir, "samples")
    tb_dir = os.path.join(run_dir, "tensorboard")
    for d in (checkpoints_dir, samples_dir, tb_dir):
        os.makedirs(d, exist_ok=True)
    print(f"Saving outputs to: {run_dir}")

    writer = SummaryWriter(log_dir=tb_dir)
    codebook_size = model.quantizer.codebook.num_embeddings

    # Train
    for epoch in range(EPOCHS):
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        total_seg_loss = 0
        total_lpips_loss = 0
        total_unique = 0
        last_batch = None
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in loop:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            pseudo_labels = get_pseudo_labels(segformer, batch, seg_mean, seg_std)
            recon, vq_loss, indices, seg_logits = model(batch)
            # LPIPS expects images in [-1, 1]
            lpips_loss = lpips_fn(recon * 2 - 1, batch * 2 - 1).mean()
            loss, recon_loss, seg_loss = loss_fn(recon, batch, vq_loss, seg_logits, pseudo_labels, lpips_loss, weight_mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_seg_loss += seg_loss.item()
            total_lpips_loss += lpips_loss.item()
            total_unique += indices.unique().numel()
            last_batch = batch
            loop.set_postfix(loss=loss.item() / len(batch))

        n_batches = len(dataloader)
        avg_loss = total_loss / len(dataset)
        avg_recon = total_recon_loss / n_batches
        avg_vq = total_vq_loss / n_batches
        avg_seg = total_seg_loss / n_batches
        avg_lpips = total_lpips_loss / n_batches
        avg_unique = total_unique / n_batches

        writer.add_scalar("loss/total", avg_loss, epoch)
        writer.add_scalar("loss/recon", avg_recon, epoch)
        writer.add_scalar("loss/vq", avg_vq, epoch)
        writer.add_scalar("loss/seg", avg_seg, epoch)
        writer.add_scalar("loss/lpips", avg_lpips, epoch)
        writer.add_scalar("codebook/unique_usage", avg_unique, epoch)
        writer.add_scalar("codebook/usage_pct", avg_unique / codebook_size * 100, epoch)

        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f} | Seg: {avg_seg:.4f} | LPIPS: {avg_lpips:.4f} | Codebook: {avg_unique:.0f}/{codebook_size}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            if last_batch is not None:
                with torch.no_grad():
                    sample = last_batch[:8]
                    recon, _, _, _ = model(sample)
                    comparison = torch.cat([sample, recon], dim=0)

                    save_image(
                        comparison,
                        os.path.join(samples_dir, f"epoch_{epoch + 1}.png"),
                        nrow=8,
                    )

                    if (epoch + 1) % 20 == 0:
                        save_seg_maps(
                            segformer, sample, recon, seg_mean, seg_std,
                            os.path.join(samples_dir, f"seg_epoch_{epoch + 1}.png"),
                        )
                        print(f"Segmentation maps saved for epoch {epoch + 1}")

    # Save final model
    final_path = os.path.join(checkpoints_dir, "final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}!")
    writer.close()


if __name__ == "__main__":
    main()
