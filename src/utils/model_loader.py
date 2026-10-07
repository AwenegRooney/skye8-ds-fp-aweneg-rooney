from pathlib import Path

import tensorflow as tf


def load_or_build_model(
    model=None,
    model_path: str | None = None,
    architecture: str = "MobileNetV2",
    input_shape=(224, 224, 3),
    weights="imagenet",
    classes=1000,
    **arch_kwargs,
) -> tf.keras.Model:
    if model is not None:
        return model

    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return tf.keras.models.load_model(path)

    # Build from applications
    arch = getattr(tf.keras.applications, architecture, None)
    if arch is None:
        raise ValueError(f"Unknown architecture: {architecture}")

    return arch(
        input_shape=input_shape,
        weights=weights,
        classes=classes,
        **arch_kwargs,
    )
