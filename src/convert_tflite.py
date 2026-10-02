"""Export the official ImageNet MobileNetV2 as FP32 TFLite."""
from pathlib import Path

import tensorflow as tf


def main(output_dir_loc: str = "models"):
    output_dir = Path(output_dir_loc)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        weights="imagenet",
        classes=1000,
    )
    model.trainable = False

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    out = output_dir / "mobilenet_v2_baseline.tflite"
    out.write_bytes(tflite_model)
    print(f"Saved Baseline → {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
