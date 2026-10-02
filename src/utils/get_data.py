from __future__ import annotations

import numpy as np
import tensorflow as tf

SAMPLES = [
    ("n02129604_tiger", 292),
    ("n02114367_timber_wolf", 269),
    ("n02124075_Egyptian_cat", 285),
    ("n02099601_golden_retriever", 207),
    ("n02123159_tiger_cat", 282),
    ("n02129165_lion", 291),
    ("n02130308_cheetah", 293),
    ("n02119022_red_fox", 277),
    ("n02086240_Shih-Tzu", 155),
    ("n02088364_beagle", 162),
    ("n04285008_sports_car", 817),
    ("n02690373_airliner", 404),
    ("n04146614_school_bus", 779),
    ("n03666591_lighthouse", 437),
    ("n02958343_car_wheel", 468),
    ("n04037443_racer", 751),
    ("n02814533_beach_wagon", 436),
    ("n03417042_garbage_truck", 555),
    ("n07753592_banana", 954),
    ("n07873807_pizza", 963),
    ("n07747607_orange", 950),
    ("n07749582_lemon", 951),
    ("n07720875_bell_pepper", 948),
    ("n07734744_mushroom", 937),
    ("n01440764_tench", 0),
    ("n01443537_goldfish", 1),
    ("n01622779_great_grey_owl", 24),
    ("n01855672_goose", 99),
    ("n02056570_king_penguin", 145),
    ("n02165456_ladybug", 301),
    ("n02165456_ladybug", 301),
    ("n02129165_lion", 291),
    ("n02129604_tiger", 292),
    ("n02099601_golden_retriever", 207),
    ("n04285008_sports_car", 817),
]

BASE_URL = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"


def _load_one(stem: str) -> np.ndarray:
    url = f"{BASE_URL}{stem}.JPEG"
    path = tf.keras.utils.get_file(
        fname=f"{stem}.jpeg",
        origin=url,
        cache_subdir="imagenet_samples",
    )
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    return arr


def load_raw_images(max_images: int = 100) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for stem, idx in SAMPLES[:max_images]:
        try:
            arr = _load_one(stem)
            images.append(arr)
            labels.append(idx)
        except Exception as e:
            print(f"Skip {stem}: {e}")
    x = np.stack(images, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Loaded {len(x)} images")
    return x, y


def preprocess(x: np.ndarray) -> np.ndarray:
    """MobileNetV2 preprocessing → [-1, 1]"""
    return tf.keras.applications.mobilenet_v2.preprocess_input(x.copy())


def representative_dataset(num_calib: int = 100):
    """Generator used by TFLiteConverter for full-integer PTQ."""
    x, _ = load_raw_images(max_images=num_calib)
    x = preprocess(x)
    for i in range(len(x)):
        yield [x[i : i + 1]]  # shape (1, 224, 224, 3)


def load_finetune_data(num_samples: int = 64):
    """Returns (x, y) already preprocessed, tiled if needed."""
    x, y = load_raw_images(max_images=40)
    x = preprocess(x)
    repeats = (num_samples // len(x)) + 1
    x = np.tile(x, (repeats, 1, 1, 1))[:num_samples]
    y = np.tile(y, repeats)[:num_samples]
    return x, y
