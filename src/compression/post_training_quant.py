from pathlib import Path

import numpy as np
import tensorflow as tf

from ..utils.get_data import load_real_sample_data
from ..utils.model_loader import load_compatible_keras_model


def post_training_quantization(base_path: Path, output_path: Path) -> None:
    base_model = load_compatible_keras_model(base_path / "mobilenet_v2_baseline.h5")
    real_data = load_real_sample_data(num_samples=100)

    def representative_data_gen():
        for sample in real_data:
            yield [sample[np.newaxis, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_quant_model = converter.convert()

    out_file = output_path / "mobilenet_v2_ptq.tflite"
    with open(out_file, "wb") as f:
        f.write(tflite_quant_model)

    print(f"Saved accuracy-preserved PTQ model to {out_file}")


if __name__ == "__main__":
    base_path = Path("models")
    output_path = Path("models")
    post_training_quantization(base_path, output_path)
