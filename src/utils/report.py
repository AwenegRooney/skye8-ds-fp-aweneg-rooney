"""Benchmarking and report generation for Phonify."""

from __future__ import annotations

import json
import platform
import resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import tensorflow as tf

from .data_loader import DatasetBundle, InputSpec, canonical_name


def directory_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def peak_rss_mb() -> float:
    """Return the current process high-water RSS in MB on Linux/macOS."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _safe_numeric_input_from_details(detail: Mapping[str, Any]) -> np.ndarray:
    shape = [int(value) if int(value) > 0 else 1 for value in detail["shape"]]
    dtype = np.dtype(detail["dtype"])
    if np.issubdtype(dtype, np.bool_):
        return np.zeros(shape, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = max(info.min, -16 if dtype == np.int8 else info.min)
        high = min(info.max, 16 if dtype == np.int8 else info.max)
        return (
            np.zeros(shape, dtype=dtype)
            if low == high
            else np.random.randint(low, high + 1, size=shape, dtype=dtype)
        )
    if np.issubdtype(dtype, np.floating):
        return np.random.uniform(-1.0, 1.0, size=shape).astype(dtype)
    if dtype.kind in {"U", "S", "O"}:
        return np.full(shape, "", dtype=dtype)
    raise ValueError(f"Unsupported TFLite input dtype {dtype}.")


def _quantize_for_tflite(array: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    target_dtype = np.dtype(detail["dtype"])
    value = np.asarray(array)
    if value.dtype == target_dtype:
        return value.astype(target_dtype, copy=False)

    quantization = detail.get("quantization", (0.0, 0))
    scale = float(quantization[0])
    zero_point = int(quantization[1])

    if target_dtype in (np.dtype(np.int8), np.dtype(np.uint8)) and scale > 0.0:
        info = np.iinfo(target_dtype)
        quantized = np.rint(value.astype(np.float32) / scale + zero_point)
        return np.clip(quantized, info.min, info.max).astype(target_dtype)

    return value.astype(target_dtype, copy=False)


def _dequantize_tflite_output(value: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if dtype not in (np.dtype(np.int8), np.dtype(np.uint8)):
        return value
    quantization = detail.get("quantization", (0.0, 0))
    scale = float(quantization[0])
    zero_point = int(quantization[1])
    if scale == 0.0:
        return value.astype(np.float32)
    return (value.astype(np.float32) - zero_point) * scale


def _map_dataset_to_details(
    details: Sequence[Mapping[str, Any]],
    bundle: DatasetBundle,
    index: int,
) -> list[np.ndarray]:
    normalized = {canonical_name(name): array for name, array in bundle.inputs.items()}
    tensors: list[np.ndarray] = []
    for detail in details:
        name = str(detail["name"])
        array = bundle.inputs.get(name)
        if array is None:
            array = normalized.get(canonical_name(name))
        if array is None:
            raise ValueError(
                f"Evaluation dataset does not contain TFLite input {name!r}. "
                f"Available inputs: {list(bundle.inputs)}"
            )
        tensors.append(_quantize_for_tflite(array[index : index + 1], detail))
    return tensors


def _create_interpreter(
    tflite_path: Path,
    *,
    num_threads: int = 1,
) -> tuple[Any, str]:
    """Create a TFLite interpreter, falling back when the default XNNPACK
    delegate cannot prepare the graph.

    TensorFlow may implicitly apply XNNPACK on supported builds. Some models
    can fail while XNNPACK prepares the delegated subgraph even though the
    normal TFLite builtin kernels can execute the model. In that case we
    recreate the interpreter with BUILTIN_WITHOUT_DEFAULT_DELEGATES.
    """
    import tensorflow as tf

    try:
        interpreter = tf.lite.Interpreter(
            model_path=str(tflite_path),
            num_threads=num_threads,
        )
        interpreter.allocate_tensors()
        return interpreter, "default_delegate"
    except (RuntimeError, ValueError) as first_error:
        # The first interpreter is unusable after AllocateTensors fails.
        # Create a fresh interpreter with TensorFlow Lite's builtin resolver
        # and without default delegates such as XNNPACK.
        try:
            interpreter = tf.lite.Interpreter(
                model_path=str(tflite_path),
                num_threads=num_threads,
                experimental_op_resolver_type=(
                    tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                ),
            )
            interpreter.allocate_tensors()
            print(
                "WARNING: Default TFLite delegate failed during host "
                "benchmarking; retried with builtin kernels without "
                "default delegates. "
                f"Original error: {first_error}"
            )
            return interpreter, "builtin_without_default_delegates"
        except (RuntimeError, ValueError) as fallback_error:
            raise RuntimeError(
                "TFLite could not allocate tensors even after disabling "
                "default delegates. The model itself may contain an op "
                "that the TFLite runtime cannot prepare. "
                f"Default-delegate error: {first_error}; "
                f"builtin-kernel error: {fallback_error}"
            ) from fallback_error


def benchmark_tflite(
    tflite_path: Path,
    input_specs: Sequence[InputSpec],
    dataset: DatasetBundle | None = None,
    runs: int = 30,
) -> dict[str, Any]:
    """Benchmark the TFLite model on the host CPU.

    This is explicitly a host benchmark. Device metrics belong to the Android
    benchmark app and must not be mislabeled as phone results.
    """
    interpreter, backend = _create_interpreter(
        tflite_path,
        num_threads=1,
    )

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if dataset is not None:
        input_samples = _map_dataset_to_details(input_details, dataset, 0)
    else:
        input_samples = [_safe_numeric_input_from_details(detail) for detail in input_details]

    for detail, array in zip(input_details, input_samples, strict=True):
        interpreter.set_tensor(detail["index"], array)
    interpreter.invoke()

    start = time.perf_counter()
    for _ in range(max(1, runs)):
        for detail, array in zip(input_details, input_samples, strict=True):
            interpreter.set_tensor(detail["index"], array)
        interpreter.invoke()
        for detail in output_details:
            _ = interpreter.get_tensor(detail["index"])
    elapsed = time.perf_counter() - start

    return {
        "avg_latency_ms": (elapsed / max(1, runs)) * 1000.0,
        "runs": max(1, runs),
        "process_peak_rss_mb": peak_rss_mb(),
        "backend": backend,
        "input_details": _serialize_tensor_details(input_details),
        "output_details": _serialize_tensor_details(output_details),
    }


def _serialize_tensor_details(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for detail in details:
        quantization = detail.get("quantization")
        result.append(
            {
                "name": str(detail.get("name", "")),
                "shape": np.asarray(detail.get("shape", [])).tolist(),
                "shape_signature": np.asarray(
                    detail.get("shape_signature", detail.get("shape", []))
                ).tolist(),
                "dtype": np.dtype(detail.get("dtype")).name,
                "quantization_scale": float(quantization[0])
                if quantization is not None and len(quantization) == 2
                else 0.0,
                "quantization_zero_point": int(quantization[1])
                if quantization is not None and len(quantization) == 2
                else 0,
            }
        )
    return result


def run_keras_predictions(
    model: tf.keras.Model, bundle: DatasetBundle, limit: int | None = None
) -> np.ndarray:
    count = bundle.size if limit is None else min(bundle.size, limit)
    arrays = [
        bundle.inputs[name][:count] for name in [spec.name for spec in _keras_input_specs(model)]
    ]
    # A list matches Keras input order for functional/multi-input models.
    outputs = model(arrays if len(arrays) > 1 else arrays[0], training=False)
    return _first_output_numpy(outputs)


def run_saved_model_predictions(
    serving_fn: Any,
    input_specs: Sequence[InputSpec],
    bundle: DatasetBundle,
    limit: int | None = None,
) -> np.ndarray:
    count = bundle.size if limit is None else min(bundle.size, limit)
    kwargs = {spec.name: bundle.inputs[spec.name][:count] for spec in input_specs}
    outputs = serving_fn(**kwargs)
    return _first_output_numpy(outputs)


def run_tflite_predictions(
    tflite_path: Path,
    bundle: DatasetBundle,
    limit: int | None = None,
) -> np.ndarray:
    interpreter, _ = _create_interpreter(
        tflite_path,
        num_threads=1,
    )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    count = bundle.size if limit is None else min(bundle.size, limit)
    predictions: list[np.ndarray] = []

    for index in range(count):
        inputs = _map_dataset_to_details(input_details, bundle, index)
        for detail, array in zip(input_details, inputs, strict=True):
            interpreter.set_tensor(detail["index"], array)
        interpreter.invoke()
        value = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(_dequantize_tflite_output(np.asarray(value), output_details[0]))

    if not predictions:
        raise ValueError("No evaluation samples were available.")
    return np.concatenate(predictions, axis=0)


def _numpy_dtype_from_tensor_dtype(dtype: Any) -> np.dtype:
    """Normalize TensorFlow/Keras dtype representations to NumPy dtype."""
    numpy_dtype = getattr(dtype, "as_numpy_dtype", None)
    if numpy_dtype is not None:
        return np.dtype(numpy_dtype)
    return np.dtype(dtype)


def _keras_input_specs(model: tf.keras.Model) -> list[InputSpec]:
    specs: list[InputSpec] = []
    for tensor in model.inputs:
        name = str(tensor.name)
        shape = tuple(None if dim is None else int(dim) for dim in tensor.shape)
        dtype = _numpy_dtype_from_tensor_dtype(tensor.dtype)
        specs.append(
            InputSpec(
                name=name,
                shape=shape,
                dtype=dtype,
            )
        )
    return specs


def _first_output_numpy(outputs: Any) -> np.ndarray:
    if isinstance(outputs, Mapping):
        if not outputs:
            raise ValueError("Model returned an empty output mapping.")
        outputs = next(iter(outputs.values()))
    elif isinstance(outputs, list | tuple):
        if not outputs:
            raise ValueError("Model returned an empty output list.")
        outputs = outputs[0]
    return np.asarray(outputs)


def infer_metric(metric: str, predictions: np.ndarray, targets: np.ndarray) -> str:
    if metric != "auto":
        return metric
    if np.issubdtype(np.asarray(targets).dtype, np.integer):
        return "accuracy"
    if predictions.ndim >= 2 and predictions.shape[-1] > 1:
        return "accuracy"
    return "mae"


def calculate_metric(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric: str = "auto",
) -> dict[str, Any]:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    selected = infer_metric(metric, predictions, targets)

    if selected == "accuracy":
        return _classification_report(predictions, targets)
    if selected == "mae":
        values = np.asarray(predictions).reshape(-1) - np.asarray(targets).reshape(-1)
        return {"metric": "mae", "value": float(np.mean(np.abs(values)))}
    if selected == "rmse":
        values = np.asarray(predictions).reshape(-1) - np.asarray(targets).reshape(-1)
        return {"metric": "rmse", "value": float(np.sqrt(np.mean(values**2)))}
    raise ValueError(f"Unsupported metric {selected!r}. Use auto, accuracy, mae, or rmse.")


def _classification_report(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predictions)
    true = np.asarray(targets)

    if pred.ndim > 1 and pred.shape[-1] > 1:
        pred_ids = np.argmax(pred, axis=-1)
    else:
        values = pred.reshape(-1)
        threshold = 0.5 if np.all((values >= 0.0) & (values <= 1.0)) else 0.0
        pred_ids = (values >= threshold).astype(np.int64)

    if true.ndim > 1 and true.shape[-1] > 1:
        true_ids = np.argmax(true, axis=-1)
    else:
        true_ids = true.reshape(-1).astype(np.int64)

    if len(pred_ids) != len(true_ids):
        raise ValueError(
            f"Prediction count {len(pred_ids)} does not match target count {len(true_ids)}."
        )

    accuracy = float(np.mean(pred_ids == true_ids)) if len(true_ids) else 0.0
    classes = sorted(set(true_ids.tolist()))
    per_class: dict[str, dict[str, Any]] = {}
    for class_id in classes:
        mask = true_ids == class_id
        class_accuracy = float(np.mean(pred_ids[mask] == true_ids[mask])) if np.any(mask) else 0.0
        per_class[str(int(class_id))] = {
            "samples": int(np.sum(mask)),
            "accuracy": class_accuracy,
        }

    return {
        "metric": "accuracy",
        "value": accuracy,
        "predicted_labels": pred_ids.tolist(),
        "true_labels": true_ids.tolist(),
        "per_class": per_class,
    }


def generate_report(
    *,
    model_name: str,
    source_path: Path,
    source_format: str,
    optimize_level: str,
    tflite_path: Path,
    input_specs: Sequence[InputSpec],
    tflite_benchmark: Mapping[str, Any],
    baseline_metric: Mapping[str, Any] | None,
    optimized_metric: Mapping[str, Any] | None,
    out_dir: Path,
    host_platform: str,
) -> Path:
    source_size_mb = directory_size_bytes(source_path) / 1e6
    tflite_size_mb = tflite_path.stat().st_size / 1e6
    compression_ratio = source_size_mb / tflite_size_mb if tflite_size_mb else 0.0

    accuracy_delta = None
    if (
        baseline_metric
        and optimized_metric
        and baseline_metric.get("metric") == "accuracy"
        and optimized_metric.get("metric") == "accuracy"
    ):
        accuracy_delta = float(optimized_metric["value"]) - float(baseline_metric["value"])

    lines: list[str] = [
        f"# Phonify Benchmark Report - {model_name}",
        "",
        f"**Source format:** `{source_format}`  ",
        f"**Optimization:** `{optimize_level.upper()}`  ",
        f"**Host:** `{host_platform}`  ",
        "",
        "## 1. Storage",
        "| Metric | Source model | TFLite | Result |",
        "|---|---:|---:|---:|",
        f"| File size | {source_size_mb:.2f} MB | {tflite_size_mb:.2f} MB | {compression_ratio:.2f}x size ratio |",
        "",
        "## 2. Inputs",
        "| Name | Shape | Dtype |",
        "|---|---|---|",
    ]
    for spec in input_specs:
        lines.append(f"| `{spec.name}` | `{list(spec.shape)}` | `{spec.dtype.name}` |")

    lines.extend(
        [
            "",
            "## 3. Host CPU benchmark",
            "This section is a host benchmark, not an Android-phone measurement.",
            "",
            f"- Average latency: **{tflite_benchmark['avg_latency_ms']:.2f} ms**",
            f"- Runs: **{tflite_benchmark['runs']}**",
            f"- Host runtime backend: **`{tflite_benchmark.get('backend', 'unknown')}`**",
            f"- Process peak RSS: **{tflite_benchmark['process_peak_rss_mb']:.2f} MB**",
            "",
        ]
    )

    if baseline_metric is not None or optimized_metric is not None:
        lines.extend(
            [
                "## 4. Accuracy / evaluation",
                "| Result | Metric | Value |",
                "|---|---|---:|",
            ]
        )
        if baseline_metric is not None:
            lines.append(
                f"| Baseline | `{baseline_metric['metric']}` | {float(baseline_metric['value']):.6f} |"
            )
        if optimized_metric is not None:
            lines.append(
                f"| Optimized | `{optimized_metric['metric']}` | {float(optimized_metric['value']):.6f} |"
            )
        if accuracy_delta is not None:
            lines.append(f"| Delta | `accuracy` | {accuracy_delta:+.6f} |")
        if optimized_metric and optimized_metric.get("metric") == "accuracy":
            lines.extend(
                ["", "### Per-class accuracy", "| Class | Samples | Accuracy |", "|---:|---:|---:|"]
            )
            baseline_classes = (baseline_metric or {}).get("per_class", {})
            for class_id, result in optimized_metric.get("per_class", {}).items():
                before = baseline_classes.get(class_id, {}).get("accuracy")
                before_text = "" if before is None else f" (baseline {before:.4f})"
                lines.append(
                    f"| {class_id} | {result['samples']} | {result['accuracy']:.4f}{before_text} |"
                )
    else:
        lines.extend(
            [
                "## 4. Accuracy / evaluation",
                "No evaluation dataset was supplied. Re-run with `--eval-data` to report accuracy or error metrics.",
            ]
        )

    lines.extend(
        [
            "",
            "## 5. Android device benchmark",
            "Android size, memory and latency must be measured by the companion app on the real target phone. "
            "Do not substitute this host benchmark for the required device measurement.",
            "",
            "### Artifact metadata",
            "```json",
            json.dumps(
                {
                    "model": model_name,
                    "optimization": optimize_level,
                    "source_format": source_format,
                    "tflite_path": str(tflite_path),
                    "tflite_size_mb": round(tflite_size_mb, 4),
                    "device_benchmark_required": True,
                },
                indent=2,
            ),
            "```",
            "",
            "*Generated automatically by Phonify.*",
        ]
    )

    report_path = out_dir / f"{model_name}_{optimize_level}_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "source_format": source_format,
                "optimization": optimize_level,
                "source_size_mb": source_size_mb,
                "tflite_size_mb": tflite_size_mb,
                "compression_ratio": compression_ratio,
                "input_specs": [
                    {"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype.name}
                    for spec in input_specs
                ],
                "host_benchmark": dict(tflite_benchmark),
                "baseline_metric": baseline_metric,
                "optimized_metric": optimized_metric,
                "device_benchmark_required": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path
