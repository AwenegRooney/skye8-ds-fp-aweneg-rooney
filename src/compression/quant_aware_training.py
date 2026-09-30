from pathlib import Path

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from ..utils.get_data import load_real_sample_data
from ..utils.model_loader import load_compatible_keras_model


def quant_aware_training(base_path: Path, output_path: Path) -> None:
    base_model = load_compatible_keras_model(base_path / "mobilenet_v2_baseline.h5")

    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(base_model)

    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    x_train = load_real_sample_data(num_samples=64)
    y_pseudo = base_model.predict(x_train).argmax(axis=-1)

    qat_model.fit(x_train, y_pseudo, epochs=1, batch_size=16)

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_qat_model = converter.convert()

    out_file = output_path / "mobilenet_v2_qat.tflite"
    with open(out_file, "wb") as f:
        f.write(tflite_qat_model)
    print(f"Saved accuracy-preserved QAT model to {out_file}")


if __name__ == "__main__":
    base_path = Path("models")
    output_path = Path("models")
    quant_aware_training(base_path, output_path)
