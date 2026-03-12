"""
Pre-compute SegFormer segmentation labels for all training images.
Saves per-image label tensors to dataset_raw/seg_labels/<stem>.pt
Run once before training: python precompute_seg.py
"""
import os
import glob
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

DATA_PATH = "dataset_raw/images/*.jpg"
OUT_DIR = "dataset_raw/seg_labels"
SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
BATCH_SIZE = 4
DEVICE = "cuda"

SEG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
SEG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(DATA_PATH))
    print(f"Found {len(files)} images. Loading SegFormer...")

    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    model = model.to(DEVICE)  # type: ignore[arg-type]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    already_done = set(os.path.splitext(f)[0] for f in os.listdir(OUT_DIR))
    todo = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in already_done]
    print(f"{len(todo)} images remaining (skipping {len(files) - len(todo)} already done).")

    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="Segmenting"):
        batch_files = todo[i:i + BATCH_SIZE]
        imgs = []
        for path in batch_files:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = torch.FloatTensor(img / 255.0).permute(2, 0, 1)
            imgs.append(img)

        batch = torch.stack(imgs).to(DEVICE)
        normalized = (batch - SEG_MEAN) / SEG_STD

        with torch.no_grad():
            logits = model(pixel_values=normalized).logits  # (B, 19, H/4, W/4)
            labels = F.interpolate(logits, size=batch.shape[-2:], mode='bilinear', align_corners=False).argmax(dim=1)  # (B, H, W)

        labels = labels.to(torch.int16).cpu()
        torch.cuda.empty_cache()
        for j, path in enumerate(batch_files):
            stem = os.path.splitext(os.path.basename(path))[0]
            torch.save(labels[j], os.path.join(OUT_DIR, f"{stem}.pt"))

    print(f"Done. Labels saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
