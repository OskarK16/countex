import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ImageListDataset(Dataset):
    """
        A list of (image path, label) pairs with a transform applied on load.
    """

    def __init__(self, samples, class_names, transform = None):
        self.samples = samples # list of tuples (image_path, label)
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _read_lines(path):
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


def load_awa2_samples(root):
    root = Path(root)
    class_names = [parts[1] for parts in _read_lines(root / "classes.txt")]

    samples = []
    for label, name in enumerate(class_names):
        class_dir = root / "JPEGImages" / name
        for path in sorted(class_dir.glob("*.jpg")):
            samples.append((path, label))

    if not samples:
        raise FileNotFoundError(f"No AwA2 images found - {root / 'JPEGImages'}")

    return samples, class_names


def split_stratified(samples, test_ratio, seed):
    """AwA2 has no official train/test split for standard classification,
    so we make a per-class random split that is reproducible given the seed.
    """

    rng = np.random.default_rng(seed)
    labels = np.array([label for _, label in samples])

    train_idx, test_idx = [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    return [samples[i] for i in sorted(train_idx)], [samples[i] for i in sorted(test_idx)]


def load_cub_splits(root):
    """Return (train_samples, test_samples, class_names) using the official CUB split."""
    root = Path(root)
    class_names = [parts[1] for parts in _read_lines(root / "classes.txt")]
    paths = {int(i): p for i, p in _read_lines(root / "images.txt")}
    labels = {int(i): int(c) - 1 for i, c in _read_lines(root / "image_class_labels.txt")}
    is_train = {int(i): t == "1" for i, t in _read_lines(root / "train_test_split.txt")}

    train, test = [], []
    for image_id in sorted(paths):
        sample = (root / "images" / paths[image_id], labels[image_id])
        (train if is_train[image_id] else test).append(sample)
    return train, test, class_names


def build_datasets(dataset, root, train_transform, eval_transform, test_ratio = 0.2, seed = 2026):
    """Return (train_dataset, test_dataset, class_names)."""

    if dataset == "awa2":
        samples, class_names = load_awa2_samples(root)
        train, test = split_stratified(samples, test_ratio, seed)
    elif dataset == "cub":
        train, test, class_names = load_cub_splits(root)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return (ImageListDataset(train, class_names, train_transform),
            ImageListDataset(test, class_names, eval_transform),
            class_names)
