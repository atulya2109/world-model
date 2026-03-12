import os
import uuid
from datetime import datetime
import cv2
import glob
import json
import lpips
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from tqdm import tqdm
from models import VQVAE

MASK_COORDS_PATH = "mask_coords.json"
PRIORITY_COORDS_PATH = "priority_coords.json"
SEG_LABELS_DIR = "dataset_raw/seg_labels"

DEVICE = "cuda"
DATA_PATH = "dataset_raw/images/*.jpg"
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PRIORITY_WEIGHT = 100.0  # How much more to weight the priority region
SEG_LOSS_WEIGHT = 0.5
LPIPS_LOSS_WEIGHT = 0.05


class DriveDataset(Dataset):
    def __init__(self, glob_pattern, mask_coords_path=MASK_COORDS_PATH, seg_labels_dir=SEG_LABELS_DIR):
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

        self.seg_labels_dir = seg_labels_dir
        has_labels = os.path.isdir(seg_labels_dir) and len(os.listdir(seg_labels_dir)) > 0
        print(f"Pre-computed seg labels: {'enabled' if has_labels else 'NOT FOUND — run precompute_seg.py first'}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        if self.mask_coords is not None:
            x1 = self.mask_coords["x1"]
            y1 = self.mask_coords["y1"]
            x2 = self.mask_coords["x2"]
            y2 = self.mask_coords["y2"]
            img[y1:y2, x1:x2] = 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = torch.FloatTensor(img).permute(2, 0, 1)

        stem = os.path.splitext(os.path.basename(path))[0]
        label_path = os.path.join(self.seg_labels_dir, f"{stem}.pt")
        seg_label = torch.load(label_path, weights_only=True).long()

        return img, seg_label


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
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # Setup
    dataset = DriveDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device=DEVICE)

    # Create priority mask (assumes all images are same size)
    sample_img, _ = dataset[0]
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

        for batch, pseudo_labels in loop:
            batch = batch.to(DEVICE)
            pseudo_labels = pseudo_labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type=DEVICE):
                recon, vq_loss, indices, seg_logits = model(batch)
                # LPIPS expects images in [-1, 1]
                lpips_loss = lpips_fn(recon * 2 - 1, batch * 2 - 1).mean()
                loss, recon_loss, seg_loss = loss_fn(recon, batch, vq_loss, seg_logits, pseudo_labels, lpips_loss, weight_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

            with torch.no_grad(), autocast(device_type=DEVICE):
                sample = last_batch[:8]  # type: ignore[index]
                recon, _, _, _ = model(sample)
                comparison = torch.cat([sample.float(), recon.float()], dim=0)
                save_image(
                    comparison,
                    os.path.join(samples_dir, f"epoch_{epoch + 1}.png"),
                    nrow=8,
                )

    # Save final model
    final_path = os.path.join(checkpoints_dir, "final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}!")
    writer.close()


if __name__ == "__main__":
    main()
