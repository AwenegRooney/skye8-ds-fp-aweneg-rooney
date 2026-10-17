"""Phonify: one-command TensorFlow/Keras -> TFLite conversion and profiling.

This version adds an important TFLite conversion step for Keras models:
resource variables are frozen into constants before the ConcreteFunction is
handed to TFLite. This prevents common READ_VARIABLE failures during INT8
representative-dataset calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from ..utils.data_loader import (
    DatasetBundle,
    InputSpec,
    canonical_name,
    load_tensor_dataset,
    representative_generator,
    resolve_dynamic_shapes,
    validate_dataset,
    validate_int8_calibration,
)
from ..utils.report import (
    benchmark_tflite,
    calculate_metric,
    generate_report,
    run_keras_predictions,
    run_saved_model_predictions,
    run_tflite_predictions,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class LoadedModel:
    def __init__(
        self,
        source_format: str,
        source_path: Path,
        keras_model: tf.keras.Model | None,
        serving_fn: Any,
        input_specs: list[InputSpec],
        converter_factory,
    ) -> None:
        self.source_format = source_format
        self.source_path = source_path
        self.keras_model = keras_model
        self.serving_fn = serving_fn
        self.input_specs = input_specs
        self.converter_factory = converter_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phonify: domain-agnostic TensorFlow/Keras to TFLite converter. "
            "Calibration/evaluation files contain already-preprocessed numeric tensors."
        )
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to .keras, .h5, or TensorFlow SavedModel directory",
    )

    parser.add_argument(
        "--optimize",
        choices=["none", "dynamic", "int8"],
        default="int8",
        help="Optimization mode: none, dynamic, or full INT8 PTQ",
    )

    parser.add_argument(
        "--calib-data",
        "--calib_data",
        dest="calib_data",
        default=None,
        help="Calibration tensors (.npy for single input, .npz for one/multiple inputs)",
    )

    parser.add_argument(
        "--eval-data",
        "--eval_data",
        dest="eval_data",
        default=None,
        help="Evaluation tensors (.npy/.npz). NPZ may contain __target__.",
    )

    parser.add_argument(
        "--labels",
        default=None,
        help="Optional .npy/.npz labels for --eval-data when labels are not stored in the dataset",
    )

    parser.add_argument(
        "--metric",
        choices=["auto", "accuracy", "mae", "rmse"],
        default="auto",
        help=(
            "Evaluation metric. `auto` uses integer labels/multi-class outputs "
            "to choose accuracy; otherwise MAE."
        ),
    )

    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        default="./phone_models",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Host CPU TFLite benchmark iterations",
    )

    parser.add_argument(
        "--calib-samples",
        type=int,
        default=100,
        help="Representative calibration sample count",
    )

    parser.add_argument(
        "--input-shape",
        action="append",
        default=[],
        metavar="NAME=D0,D1,...",
        help="Resolve a dynamic input shape explicitly. Repeat for multiple inputs.",
    )

    parser.add_argument(
        "--signature-key",
        default="serving_default",
        help="SavedModel signature used for conversion/evaluation",
    )

    return parser.parse_args()


def parse_explicit_shapes(values: list[str]) -> dict[str, tuple[int, ...]]:
    """Parse repeated NAME=D0,D1,... command-line shape specifications."""
    result: dict[str, tuple[int, ...]] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --input-shape {value!r}. Expected NAME=D0,D1,...")

        name, raw_shape = value.split("=", 1)
        name = name.strip()

        try:
            dims = tuple(int(part.strip()) for part in raw_shape.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid --input-shape {value!r}: dimensions must be integers."
            ) from exc

        if not name or not dims or any(dim <= 0 for dim in dims):
            raise ValueError(f"Invalid --input-shape {value!r}.")

        result[name] = dims
        result[canonical_name(name)] = dims

    return result


def _numpy_dtype(tensor: Any) -> np.dtype:
    """Convert a TensorFlow dtype-bearing tensor to a NumPy dtype."""
    try:
        return np.dtype(tensor.dtype.as_numpy_dtype)
    except AttributeError:
        return np.dtype(tensor.dtype)


def load_model(source: Path, signature_key: str) -> LoadedModel:
    """Load either a Keras model file or a TensorFlow SavedModel."""

    # ------------------------------------------------------------------
    # TensorFlow SavedModel directory
    # ------------------------------------------------------------------
    if source.is_dir():
        saved_model_path = source

        loaded = tf.saved_model.load(str(saved_model_path))
        signatures = loaded.signatures

        if signature_key not in signatures:
            raise ValueError(
                f"SavedModel signature {signature_key!r} not found. "
                f"Available signatures: {list(signatures)}"
            )

        serving_fn = signatures[signature_key]
        positional, kwargs = serving_fn.structured_input_signature

        if kwargs:
            specs = [
                InputSpec(
                    name=name,
                    shape=tuple(
                        None if dim is None or int(dim) < 0 else int(dim) for dim in tensor.shape
                    ),
                    dtype=np.dtype(tensor.dtype.as_numpy_dtype),
                )
                for name, tensor in kwargs.items()
            ]
        else:
            if positional:
                raise ValueError(
                    "Positional-only SavedModel signatures are not supported. "
                    "Export a named signature so calibration tensors can be matched "
                    "to model inputs by name."
                )

            raise ValueError(f"SavedModel signature {signature_key!r} has no tensor inputs.")

        return LoadedModel(
            source_format="saved_model",
            source_path=source,
            keras_model=None,
            serving_fn=serving_fn,
            input_specs=specs,
            converter_factory=lambda: tf.lite.TFLiteConverter.from_saved_model(
                str(saved_model_path),
                signature_keys=[signature_key],
            ),
        )

    # ------------------------------------------------------------------
    # Keras .keras / .h5 / .hdf5
    # ------------------------------------------------------------------
    suffix = source.suffix.lower()

    if suffix not in {".keras", ".h5", ".hdf5"}:
        raise ValueError(
            f"Unsupported model path {source}. Supported inputs are .keras, "
            ".h5/.hdf5, or a SavedModel directory."
        )

    try:
        model = tf.keras.models.load_model(
            source,
            compile=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Keras model {source}. "
            "If this is a Keras 3 SavedModel directory, pass the SavedModel "
            "directory directly instead of using load_model()."
        ) from exc

    if not getattr(model, "inputs", None):
        raise ValueError(
            "Loaded Keras model has no declared input tensors. "
            "The Stage C converter requires an explicit input contract."
        )

    specs = [
        InputSpec(
            name=str(tensor.name),
            shape=tuple(None if dim is None else int(dim) for dim in tensor.shape),
            dtype=_numpy_dtype(tensor),
        )
        for tensor in model.inputs
    ]

    return LoadedModel(
        source_format="keras",
        source_path=source,
        keras_model=model,
        serving_fn=None,
        input_specs=specs,
        converter_factory=lambda: tf.lite.TFLiteConverter.from_keras_model(model),
    )


def _require_frozen_dim(value: int | None, *, spec_name: str) -> int:
    if value is None:
        raise ValueError(f"Input {spec_name!r} still has an unresolved dimension for TensorSpec.")
    return int(value)


def _build_keras_concrete_function(
    model: tf.keras.Model,
    resolved_specs: list[InputSpec],
) -> Any:
    """Build a fixed-shape Keras inference graph and freeze its variables.

    TFLite INT8 calibration executes the converted graph with a representative
    dataset. Some TensorFlow/Keras combinations leave resource-variable reads in
    that graph; the calibration interpreter can then fail with:

        READ_VARIABLE: variable != nullptr was not true

    We avoid that failure by converting model variables into constants before
    giving the graph to TFLite.
    """

    input_signature = [
        tf.TensorSpec(
            shape=tuple(_require_frozen_dim(dim, spec_name=spec.name) for dim in spec.shape),
            dtype=tf.as_dtype(spec.dtype),
            name=canonical_name(spec.name),
        )
        for spec in resolved_specs
    ]

    @tf.function(input_signature=input_signature)
    def serving(*inputs):
        model_inputs = inputs[0] if len(inputs) == 1 else list(inputs)
        return model(model_inputs, training=False)

    concrete_fn = serving.get_concrete_function()

    # CRITICAL FIX:
    # Convert resource variables into graph constants.
    frozen_fn = convert_variables_to_constants_v2(concrete_fn)

    remaining_variables = tuple(getattr(frozen_fn, "variables", ()))

    if remaining_variables:
        raise RuntimeError(
            "The inference graph still contains resource variables after "
            f"freezing ({len(remaining_variables)} variable(s) remain)."
        )

    return frozen_fn


def _freeze_saved_model_function(
    serving_fn: Any,
) -> Any:
    """Freeze a SavedModel serving ConcreteFunction before TFLite conversion."""

    frozen_fn = convert_variables_to_constants_v2(serving_fn)

    remaining_variables = tuple(getattr(frozen_fn, "variables", ()))

    if remaining_variables:
        raise RuntimeError(
            "The SavedModel serving graph still contains resource variables "
            f"after freezing ({len(remaining_variables)} variable(s) remain)."
        )

    return frozen_fn


def _build_frozen_concrete_function(
    loaded: LoadedModel,
    resolved_specs: list[InputSpec],
) -> Any:
    """Return a frozen, fixed-shape ConcreteFunction for TFLite conversion."""

    if loaded.source_format == "keras":
        assert loaded.keras_model is not None

        return _build_keras_concrete_function(
            loaded.keras_model,
            resolved_specs,
        )

    if loaded.source_format == "saved_model":
        return _freeze_saved_model_function(loaded.serving_fn)

    raise ValueError(f"Unsupported loaded model format: {loaded.source_format}")


def configure_converter(
    loaded: LoadedModel,
    optimize: str,
    calibration: DatasetBundle | None,
    calib_samples: int,
    resolved_specs: list[InputSpec],
):
    """Configure a TFLite converter from a frozen inference graph."""

    # ------------------------------------------------------------------
    # CRITICAL FIX:
    # Build and freeze the inference ConcreteFunction before TFLite sees it.
    # This eliminates resource-variable READ_VARIABLE failures during INT8
    # representative-dataset calibration for affected TF/Keras versions.
    # ------------------------------------------------------------------
    print("      -> Freezing model variables for TFLite conversion...")

    try:
        frozen_fn = _build_frozen_concrete_function(
            loaded,
            resolved_specs,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not freeze the model inference graph before TFLite "
            "conversion. This is required for reliable INT8 calibration "
            "with resource-variable Keras/TensorFlow models."
        ) from exc

    print("      -> Resource variables frozen successfully.")

    # Convert the frozen graph directly.
    # Do NOT pass the original Keras model as the trackable object because
    # the graph has already been frozen into constants.
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_fn])

    if optimize == "none":
        return converter

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if optimize == "dynamic":
        return converter

    if optimize != "int8":
        raise ValueError(f"Unsupported optimization mode: {optimize}")

    # ------------------------------------------------------------------
    # Full INT8 PTQ requires real representative calibration data.
    # Never silently fall back to synthetic tensors here.
    # ------------------------------------------------------------------
    if calibration is None:
        raise ValueError(
            "Full INT8 PTQ requires --calib-data. "
            "Provide representative tensors matching every model input."
        )

    validate_int8_calibration(
        calibration,
        resolved_specs,
    )

    converter.representative_dataset = lambda: representative_generator(
        calibration,
        resolved_specs,
        num_samples=calib_samples,
    )

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Only force INT8 I/O when all declared model inputs are floating point.
    if all(np.issubdtype(spec.dtype, np.floating) for spec in resolved_specs):
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    return converter


def make_artifact_metadata(
    loaded: LoadedModel,
    resolved_specs: list[InputSpec],
    optimize: str,
    tflite_path: Path,
) -> dict[str, Any]:
    """Create metadata consumed by later Android/device benchmarking steps."""

    return {
        "phonify_version": "0.3.0",
        "tensorflow_version": tf.__version__,
        "source_format": loaded.source_format,
        "source_path": str(loaded.source_path),
        "optimization": optimize,
        "conversion_graph": "frozen_concrete_function",
        "inputs": [
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "dtype": spec.dtype.name,
            }
            for spec in resolved_specs
        ],
        "tflite_path": str(tflite_path),
        "tflite_size_bytes": tflite_path.stat().st_size,
        "host_benchmark_is_not_device_benchmark": True,
        "device_benchmark_required": True,
    }


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.runs <= 0:
        raise ValueError("--runs must be greater than zero.")

    if args.calib_samples <= 0:
        raise ValueError("--calib-samples must be greater than zero.")


def _print_model_summary(
    source: Path, loaded: LoadedModel, resolved_specs: list[InputSpec]
) -> None:
    print(f"\n[1/4] Loading model: {source}")
    print(f"      format: {loaded.source_format}")
    print(f"      TensorFlow: {tf.__version__}")
    print(f"      Python: {sys.version.split()[0]}")
    print("[2/4] Input contract:")
    for spec in resolved_specs:
        print(f"      - {spec.name}: shape={list(spec.shape)}, dtype={spec.dtype.name}")


def _resolve_run_data(
    args: argparse.Namespace,
    loaded: LoadedModel,
    explicit_shapes: dict[str, tuple[int, ...]],
) -> tuple[DatasetBundle | None, DatasetBundle | None, list[InputSpec]]:
    calibration: DatasetBundle | None = None
    if args.calib_data:
        calibration = load_tensor_dataset(
            args.calib_data,
            loaded.input_specs,
        )
        validate_dataset(
            calibration,
            loaded.input_specs,
        )

    evaluation: DatasetBundle | None = None
    if args.eval_data:
        evaluation = load_tensor_dataset(
            args.eval_data,
            loaded.input_specs,
            labels_path=args.labels,
        )
        validate_dataset(
            evaluation,
            loaded.input_specs,
            require_targets=True,
        )

    shape_source = calibration or evaluation
    resolved_specs = resolve_dynamic_shapes(
        loaded.input_specs,
        bundle=shape_source,
        explicit_shapes=explicit_shapes,
    )

    if calibration is not None:
        validate_dataset(calibration, resolved_specs)
    if evaluation is not None:
        validate_dataset(evaluation, resolved_specs, require_targets=True)

    return calibration, evaluation, resolved_specs


def main() -> int:
    args = parse_args()
    _validate_cli_args(args)

    source = Path(args.model_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Model path does not exist: {source}")

    explicit_shapes = parse_explicit_shapes(args.input_shape)
    loaded = load_model(source, args.signature_key)
    model_name = source.stem if source.is_file() else source.name

    calibration, evaluation, resolved_specs = _resolve_run_data(args, loaded, explicit_shapes)
    _print_model_summary(source, loaded, resolved_specs)

    converter = configure_converter(
        loaded,
        args.optimize,
        calibration,
        args.calib_samples,
        resolved_specs,
    )

    print(f"[3/4] Converting to TFLite ({args.optimize.upper()})...")

    try:
        tflite_model = converter.convert()
    except Exception as exc:
        error_text = str(exc)
        if "READ_VARIABLE" in error_text or "ReadVariable" in error_text:
            raise RuntimeError(
                "TFLite failed during calibration with READ_VARIABLE "
                "even after variable freezing. This usually indicates "
                "a TensorFlow/Keras/TFLite compatibility issue or an "
                "unsupported resource-variable/custom-op pattern in "
                "the model.\n\n"
                "Try --optimize dynamic to separate general TFLite "
                "conversion from INT8 calibration, then inspect the "
                "TensorFlow and Keras versions printed above."
            ) from exc

        raise RuntimeError(
            "TFLite conversion failed. Confirm that the model is "
            "TFLite-convertible and that calibration tensors match "
            "every input in name/order, shape, and dtype."
        ) from exc

    tflite_path = out_dir / (f"{model_name}_{args.optimize}.tflite")
    tflite_path.write_bytes(tflite_model)
    print(f"      -> {tflite_path}")

    print("[4/4] Benchmarking host CPU and generating report...")
    benchmark_dataset = evaluation or calibration
    host_benchmark = benchmark_tflite(
        tflite_path,
        resolved_specs,
        dataset=benchmark_dataset,
        runs=args.runs,
    )

    baseline_metric: Mapping[str, Any] | None = None
    optimized_metric: Mapping[str, Any] | None = None
    if evaluation is not None:
        if loaded.keras_model is not None:
            baseline_predictions = run_keras_predictions(loaded.keras_model, evaluation)
        else:
            baseline_predictions = run_saved_model_predictions(
                loaded.serving_fn,
                resolved_specs,
                evaluation,
            )

        optimized_predictions = run_tflite_predictions(tflite_path, evaluation)
        baseline_metric = calculate_metric(baseline_predictions, evaluation.targets, args.metric)
        optimized_metric = calculate_metric(optimized_predictions, evaluation.targets, args.metric)

    report_path = generate_report(
        model_name=model_name,
        source_path=source,
        source_format=loaded.source_format,
        optimize_level=args.optimize,
        tflite_path=tflite_path,
        input_specs=resolved_specs,
        tflite_benchmark=host_benchmark,
        baseline_metric=baseline_metric,
        optimized_metric=optimized_metric,
        out_dir=out_dir,
        host_platform=(f"{platform.system()} {platform.machine()}"),
    )

    metadata_path = out_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            make_artifact_metadata(
                loaded,
                resolved_specs,
                args.optimize,
                tflite_path,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"TFLite model : {tflite_path}")
    print(f"Report       : {report_path}")
    print(f"Metadata     : {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"\n[Phonify ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
