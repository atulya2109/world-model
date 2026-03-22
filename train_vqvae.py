import argparse
import os
import uuid
from datetime import datetime
import cv2
import glob
import json
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from tqdm import tqdm
from models.vae import VQVAE

MASK_COORDS_PATH = "mask_coords.json"
SEG_LABELS_DIR = "dataset_raw/seg_labels"

DEVICE = "cuda"
DATA_PATH = "dataset_raw/images/*.jpg"
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
SEG_LOSS_WEIGHT = 0.5
CLASS_VAR_PENALTY = [
    0.0,  # 0  road          — critical, use as many codes as needed
    0.05, # 1  sidewalk
    0.2,  # 2  building
    0.3,  # 3  wall
    0.3,  # 4  fence
    0.1,  # 5  pole
    0.0,  # 6  traffic light — critical
    0.0,  # 7  traffic sign  — critical
    0.5,  # 8  vegetation    — unimportant
    0.3,  # 9  terrain
    0.5,  # 10 sky           — unimportant
    0.0,  # 11 person        — critical
    0.0,  # 12 rider         — critical
    0.0,  # 13 car           — critical
    0.0,  # 14 truck         — critical
    0.0,  # 15 bus           — critical
    0.1,  # 16 train
    0.0,  # 17 motorcycle    — critical
    0.0,  # 18 bicycle       — critical
]


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


def loss_fn(recon_x, x, vq_loss, seg_logits, pseudo_labels):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    seg_loss = F.cross_entropy(seg_logits, pseudo_labels)
    total_loss = recon_loss + vq_loss * x.size(0) + SEG_LOSS_WEIGHT * seg_loss
    return total_loss, recon_loss, seg_loss



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a VQVAE checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to resume from (for logging continuity)")
    args = parser.parse_args()

    print(f"Training on {DEVICE}...")

    # Setup
    dataset = DriveDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model = VQVAE().to(DEVICE)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
        print(f"Resumed from checkpoint: {args.checkpoint}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device=DEVICE)

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
    var_penalty_tensor = torch.tensor(CLASS_VAR_PENALTY, device=DEVICE)

    # Train
    for epoch in range(args.start_epoch, args.start_epoch + EPOCHS):
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        total_seg_loss = 0
        total_var_loss = 0
        total_unique = 0
        total_perplexity = 0
        last_batch = None
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.start_epoch + EPOCHS}")

        for batch, pseudo_labels in loop:
            batch = batch.to(DEVICE)
            pseudo_labels = pseudo_labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type=DEVICE):
                recon, vq_loss, indices, seg_logits, z_e = model(batch)
                loss, recon_loss, seg_loss = loss_fn(recon, batch, vq_loss, seg_logits, pseudo_labels)

                batch_var_loss = torch.zeros(1, device=DEVICE)
                labels_latent = F.interpolate(
                    pseudo_labels.float().unsqueeze(1), size=z_e.shape[-2:], mode='nearest'
                ).squeeze(1).long()  # (B, H, W)
                z_e_hwd = z_e.permute(0, 2, 3, 1)  # (B, H, W, D)
                for cls_idx in labels_latent.unique():
                    penalty = var_penalty_tensor[cls_idx]
                    if penalty == 0.0:
                        continue
                    cls_latents = z_e_hwd[labels_latent == cls_idx]  # (N_cls, D)
                    if cls_latents.size(0) < 2:
                        continue
                    batch_var_loss = batch_var_loss + penalty * cls_latents.var(dim=0).mean()
                loss = loss + batch_var_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_seg_loss += seg_loss.item()
            total_var_loss += batch_var_loss.item()
            idx_cpu = indices.view(-1).cpu()
            total_unique += idx_cpu.unique().numel()
            counts = torch.bincount(idx_cpu, minlength=codebook_size).float()
            probs = counts / counts.sum()
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            total_perplexity += perplexity.item()
            last_batch = batch
            loop.set_postfix(loss=loss.item() / len(batch))

        n_batches = len(dataloader)
        avg_loss = total_loss / len(dataset)
        avg_recon = total_recon_loss / n_batches
        avg_vq = total_vq_loss / n_batches
        avg_seg = total_seg_loss / n_batches
        avg_var = total_var_loss / n_batches
        avg_unique = total_unique / n_batches
        avg_perplexity = total_perplexity / n_batches

        writer.add_scalar("loss/total", avg_loss, epoch)
        writer.add_scalar("loss/recon", avg_recon, epoch)
        writer.add_scalar("loss/vq", avg_vq, epoch)
        writer.add_scalar("loss/seg", avg_seg, epoch)
        writer.add_scalar("loss/var", avg_var, epoch)
        writer.add_scalar("codebook/unique_usage", avg_unique, epoch)
        writer.add_scalar("codebook/usage_pct", avg_unique / codebook_size * 100, epoch)
        writer.add_scalar("codebook/perplexity", avg_perplexity, epoch)
        writer.add_scalar("codebook/perplexity_pct", avg_perplexity / codebook_size * 100, epoch)

        mem_alloc = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        writer.add_scalar("cuda/allocated_mb", mem_alloc, epoch)
        writer.add_scalar("cuda/reserved_mb", mem_reserved, epoch)

        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f} | Seg: {avg_seg:.4f} | Var: {avg_var:.4f} | Codebook: {avg_unique:.0f}/{codebook_size} | Perplexity: {avg_perplexity:.1f}/{codebook_size} | CUDA: {mem_alloc:.0f}/{mem_reserved:.0f} MB (alloc/reserved)")

        if (epoch + 1) % 10 == 0 or epoch == args.start_epoch + EPOCHS - 1:
            checkpoint_path = os.path.join(checkpoints_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            with torch.no_grad(), autocast(device_type=DEVICE):
                sample = last_batch[:8]  # type: ignore[index]
                recon, _, _, _, _ = model(sample)
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
