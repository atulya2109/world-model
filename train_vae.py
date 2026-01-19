from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import cv2
import glob
import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models import VAE

MASK_COORDS_PATH = "mask_coords.json"
PRIORITY_COORDS_PATH = "priority_coords.json"

DEVICE = "mps"
DATA_PATH = "dataset_raw/images/*.jpg"
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
PRIORITY_WEIGHT = 5.0  # How much more to weight the priority region


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


def loss_fn(recon_x, x, mu, logvar, weight_mask=None):
    if weight_mask is not None:
        mse_per_pixel = (recon_x - x) ** 2
        weighted_mse = mse_per_pixel * weight_mask
        MSE = weighted_mse.sum()
    else:
        MSE = F.mse_loss(recon_x, x, reduction="sum")

    kld_weight = 0.5
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD * kld_weight


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

    # Setup
    dataset = DriveDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create priority mask (assumes all images are same size)
    sample_img = dataset[0]
    _, height, width = sample_img.shape
    weight_mask = create_priority_mask(
        height, width, priority_coords, PRIORITY_WEIGHT, DEVICE
    )

    # Train
    for epoch in range(EPOCHS):
        total_loss = 0
        last_batch = None
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in loop:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            recon, mu, logvar = model(batch)
            loss = loss_fn(recon, batch, mu, logvar, weight_mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            last_batch = batch
            loop.set_postfix(loss=loss.item() / len(batch))

        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(dataset):.2f}")

        if last_batch is not None:
            with torch.no_grad():
                sample = last_batch[:8]
                recon, _, _ = model(sample)
                comparison = torch.cat([sample, recon], dim=0)

                save_image(comparison, f"vae_epoch_{epoch + 1}.png", nrow=8)

    # Save Model
    torch.save(model.state_dict(), "vae.pth")
    print("Model saved to vae.pth!")


if __name__ == "__main__":
    main()
