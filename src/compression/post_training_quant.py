from pathlib import Path

import tensorflow as tf

from ..utils.get_data import representative_dataset


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
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(num_calib=120)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out = output_dir / "mobilenet_v2_ptq.tflite"
    out.write_bytes(tflite_model)
    print(f"Saved PTQ → {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
