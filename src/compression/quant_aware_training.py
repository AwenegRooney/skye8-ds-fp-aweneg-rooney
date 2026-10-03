from pathlib import Path

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from ..utils.get_data import load_finetune_data, representative_dataset


def main(output_dir_loc: str = "models", epochs: int = 3):
    output_dir = Path(output_dir_loc)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        weights="imagenet",
        classes=1000,
    )

    # Quantization-aware model
    qat_model = tfmot.quantization.keras.quantize_model(base)
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    x, y = load_finetune_data(num_samples=3000)
    qat_model.fit(x, y, epochs=epochs, batch_size=8, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(num_calib=200)
    tflite_model = converter.convert()

    out = output_dir / "mobilenet_v2_qat.tflite"
    out.write_bytes(tflite_model)
    print(f"Saved QAT → {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
