from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def _candidate_model_paths(model_path: Path) -> list[Path]:
    if model_path.suffix:
        candidates = [model_path]
    else:
        candidates = []

    candidates.extend(
        [
            model_path.with_suffix(".keras"),
            model_path.with_suffix(".h5"),
            model_path.parent / "mobilenet_v2_baseline.keras",
            model_path.parent / "mobilenet_v2_baseline.h5",
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def load_compatible_keras_model(model_path: str | Path) -> tf.keras.Model:
    """Load a Keras model safely across Keras 2/3 serialization formats."""
    model_path = Path(model_path)
    if model_path.suffix == "":
        model_path = model_path / "mobilenet_v2_baseline.keras"

    candidates = _candidate_model_paths(model_path)

    # 1. Try standard load_model first
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return tf.keras.models.load_model(candidate, compile=False)
        except Exception:
            continue

    # 2. If load_model failed due to config errors, reconstruct MobileNetV2 architecture
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights=None,
        classes=1000,
    )

    # 3. Try loading weights from existing candidate .h5/.keras files on disk
    loaded_local_weights = False
    for candidate in candidates:
        if candidate.exists():
            try:
                base_model.load_weights(candidate)
                loaded_local_weights = True
                print(f"Successfully loaded existing weights from: {candidate}")
                break
            except Exception:
                continue

    # 4. Only download ImageNet weights as a last resort if no local weights could be loaded
    if not loaded_local_weights:
        print("No valid local weights found. Falling back to ImageNet weights...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights="imagenet",
        )

    # Save reconstructed model in current format to prevent future deserialization errors
    save_path = (
        model_path
        if model_path.suffix.lower() in {".keras", ".h5"}
        else model_path.parent / "mobilenet_v2_baseline.keras"
    )
    base_model.save(save_path)
    return base_model
