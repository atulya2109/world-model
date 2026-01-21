import torch
import argparse
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, latents, actions, seq_len=32):
        self.latents = latents
        self.actions = actions
        self.seq_len = seq_len

    def __len__(self):
        return len(self.latents) - self.seq_len - 1


def main():
    parser = argparse.ArgumentParser(
        prog="DreamVisualizer", description="Generates a dream sequence"
    )
    parser.add_argument("--device", type=str, default="cuda")
