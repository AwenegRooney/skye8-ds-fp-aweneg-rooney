from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

DEFAULT_DATASET_ROOT = Path("ImageNet-Mini")


def _load_class_index(json_path: Path) -> dict[str, int]:
    """
    imagenet_class_index.json format:
      {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}

    Returns: { "n01440764": 0, "n01443537": 1, ... }
    """
    with open(json_path) as f:
        data = json.load(f)

    wnid_to_idx = {}
    for idx_str, (wnid, _name) in data.items():
        wnid_to_idx[wnid] = int(idx_str)
    return wnid_to_idx


def _collect_image_paths(
    images_dir: Path,
    wnid_to_idx: dict[str, int],
    max_total: int | None = None,
) -> list[tuple[Path, int]]:
    """Return list of (image_path, class_index)."""
    pairs = []
    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        wnid = class_dir.name
        if wnid not in wnid_to_idx:
            continue
        label = wnid_to_idx[wnid]

        files: list[Any] = []
        for ext in ("*.JPEG", "*.jpeg"):
            files.extend(class_dir.glob(ext))
        files = sorted(files)

        for f in files:
            pairs.append((f, label))
            if max_total is not None and len(pairs) >= max_total:
                return pairs
    return pairs


def load_raw_images(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    max_total: int | None = 400,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(dataset_root)
    json_path = root / "imagenet_class_index.json"
    images_dir = root / "images"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing {images_dir}")

    wnid_to_idx = _load_class_index(json_path)
    pairs = _collect_image_paths(
        images_dir,
        wnid_to_idx,
        max_total=max_total,
    )

    if len(pairs) == 0:
        raise RuntimeError("No images found. Check folder structure.")

    images, labels = [], []
    for path, label in pairs:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(image_size, Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)  # 0-255
            images.append(arr)
            labels.append(label)
        except Exception as e:
            print(f"Skip {path.name}: {e}")

    x = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    print(
        f"Loaded {len(x)} images from {dataset_root}  " f"({len(set(y.tolist()))} unique classes)"
    )
    return x, y


def preprocess(x: np.ndarray) -> np.ndarray:
    return tf.keras.applications.mobilenet_v2.preprocess_input(x.copy())


def representative_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    num_calib: int = 200,
):
    x, _ = load_raw_images(
        dataset_root=dataset_root,
        max_total=num_calib,
    )
    x = preprocess(x)
    for i in range(len(x)):
        yield [x[i : i + 1]]


def load_finetune_data(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    num_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (x, y) already preprocessed to [-1, 1].
    Set num_samples=None to load every image (watch your RAM!).
    """
    x, y = load_raw_images(
        dataset_root=dataset_root,
        max_total=num_samples,
    )
    x = preprocess(x)
    return x, y


def get_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    num_samples: int | None = None,  # None = use ALL
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Preferred way for fine-tuning.
    Streams from disk-friendly pipeline; does not keep 3000 images in RAM
    if you later switch to a pure path-based pipeline.
    """
    x, y = load_finetune_data(dataset_root=dataset_root, num_samples=num_samples)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 1024), reshuffle_each_iteration=True)
    ds = ds.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
