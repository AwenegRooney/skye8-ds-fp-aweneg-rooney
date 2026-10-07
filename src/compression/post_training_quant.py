from pathlib import Path
from typing import Optional, Union

import tensorflow as tf

from ..utils.get_data import representative_dataset
from ..utils.model_loader import load_or_build_model


def main(
    model=None,
    model_path: Optional[str] = None,
    architecture: str = "MobileNetV2",
    output_dir: Union[str, Path] = "models",
    output_name: str = "model_ptq",
    num_calib: int = 500,
    **arch_kwargs,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = load_or_build_model(
        model=model,
        model_path=model_path,
        architecture=architecture,
        **arch_kwargs,
    )
    model.trainable = False

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(num_calib=num_calib)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out = output_path / f"{output_name}.tflite"
    out.write_bytes(tflite_model)
    print(f"Saved PTQ → {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return out


if __name__ == "__main__":
    main(architecture="MobileNetV3Small", output_name="mobilenet_v3_small_ptq")
