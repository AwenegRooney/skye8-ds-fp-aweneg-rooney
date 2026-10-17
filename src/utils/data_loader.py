"""Dataset loading and strict tensor validation for Phonify.

The important design rule is that Phonify never guesses modality-specific
preprocessing. Calibration/evaluation data must already be tensors in the
format expected by the model. This makes the tool domain-agnostic without
silently applying the wrong image/audio/text preprocessing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TARGET_KEYS = ("__target__", "target", "targets", "label", "labels", "y")


@dataclass(frozen=True)
class InputSpec:
    """Resolved model input description."""

    name: str
    shape: tuple[int | None, ...]
    dtype: np.dtype

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass
class DatasetBundle:
    """A tensor dataset keyed by model input name, plus optional targets."""

    inputs: dict[str, np.ndarray]
    targets: np.ndarray | None = None

    @property
    def size(self) -> int:
        sizes = {int(value.shape[0]) for value in self.inputs.values() if value.ndim > 0}
        if not sizes:
            raise ValueError("Dataset contains no batched tensors.")
        if len(sizes) != 1:
            raise ValueError(f"Input tensors contain different sample counts: {sorted(sizes)}")
        return next(iter(sizes))


def canonical_name(name: str) -> str:
    """Normalize names for tolerant matching while preserving exact names when possible."""
    value = str(name).strip()
    if value.endswith(":0"):
        value = value[:-2]
    return value.replace("/", "_").replace(" ", "_")


def _as_tuple_shape(shape: Sequence[object]) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for dim in shape:
        if dim is None:
            result.append(None)
            continue
        try:
            if isinstance(dim, int | np.integer):
                value = int(dim)
            else:
                value = int(str(dim))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Unsupported model dimension {dim!r}") from exc
        result.append(None if value < 0 else value)
    return tuple(result)


def _load_npy_dataset(
    path: str | Path,
    input_specs: Sequence[InputSpec],
    labels_path: str | Path | None,
) -> DatasetBundle:
    if len(input_specs) != 1:
        raise ValueError(
            "A .npy dataset is supported only for a single-input model. Use .npz for multiple inputs."
        )
    array = np.load(path, allow_pickle=False)
    if array.ndim == 0:
        array = np.expand_dims(array, 0)
    bundle = DatasetBundle(inputs={input_specs[0].name: array})
    if labels_path is not None:
        bundle.targets = _load_labels(labels_path)
    validate_dataset(bundle, input_specs, require_targets=False)
    return bundle


def _map_npz_inputs(
    archive: np.lib.npyio.NpzFile,
    input_specs: Sequence[InputSpec],
) -> tuple[dict[str, np.ndarray], list[str]]:
    keys = list(archive.keys())
    target_name = next((key for key in _TARGET_KEYS if key in archive), None)
    input_keys = [key for key in keys if key != target_name and key not in _TARGET_KEYS]
    normalized_map = {canonical_name(spec.name): spec.name for spec in input_specs}
    mapped: dict[str, np.ndarray] = {}

    for spec in input_specs:
        if spec.name in archive:
            mapped[spec.name] = archive[spec.name]

    for key in input_keys:
        if key in mapped:
            continue
        target_name_for_key = normalized_map.get(canonical_name(key))
        if target_name_for_key is not None and target_name_for_key not in mapped:
            mapped[target_name_for_key] = archive[key]

    if len(input_specs) == 1 and len(mapped) == 0 and len(input_keys) == 1:
        mapped[input_specs[0].name] = archive[input_keys[0]]

    missing = [spec.name for spec in input_specs if spec.name not in mapped]
    if missing:
        raise ValueError(
            "Dataset is missing model inputs: "
            + ", ".join(missing)
            + f". Available keys: {input_keys}. Model inputs: {[spec.name for spec in input_specs]}"
        )
    return mapped, input_keys


def _load_npz_dataset(
    path: str | Path,
    input_specs: Sequence[InputSpec],
    labels_path: str | Path | None,
    target_key: str,
) -> DatasetBundle:
    archive = np.load(path, allow_pickle=False)
    try:
        target_candidates = [target_key, *_TARGET_KEYS]
        target_name = next((key for key in target_candidates if key in archive), None)
        mapped, _ = _map_npz_inputs(archive, input_specs)

        targets = archive[target_name] if target_name is not None else None
        if labels_path is not None:
            if targets is not None:
                raise ValueError(
                    "Both an NPZ target key and --labels were supplied. Provide only one."
                )
            targets = _load_labels(labels_path)

        bundle = DatasetBundle(inputs=mapped, targets=targets)
        validate_dataset(bundle, input_specs, require_targets=False)
        return bundle
    finally:
        archive.close()


def load_tensor_dataset(
    path: str | Path,
    input_specs: Sequence[InputSpec],
    labels_path: str | Path | None = None,
    target_key: str = "__target__",
) -> DatasetBundle:
    """Load .npy or .npz tensors and map them to model input names.

    Recommended .npz contract:
      - one key per model input, using the model input name
      - optional target key `__target__`

    .npy is supported only for a single-input model. The optional labels file
    is used as the evaluation target for .npy evaluation datasets.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_dir():
        raise ValueError(
            "Directory datasets are intentionally not auto-preprocessed. "
            "Convert raw image/audio/text data into .npy/.npz tensors first, "
            "so Phonify does not guess the model's preprocessing."
        )

    if path.suffix.lower() == ".npy":
        return _load_npy_dataset(path, input_specs, labels_path)

    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported dataset format {path.suffix!r}. Use .npy or .npz.")

    return _load_npz_dataset(path, input_specs, labels_path, target_key)


def _load_labels(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labels path does not exist: {path}")
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            keys = list(archive.keys())
            target_key = next(
                (
                    key
                    for key in ["__target__", "target", "targets", "label", "labels", "y"]
                    if key in archive
                ),
                None,
            )
            if target_key is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"Labels NPZ must contain one array or a known target key. Found: {keys}"
                    )
                target_key = keys[0]
            return archive[target_key].copy()
        finally:
            archive.close()
    raise ValueError("Labels must be a .npy or .npz file.")


def _validate_dataset_shapes(bundle: DatasetBundle, input_specs: Sequence[InputSpec]) -> int:
    size = bundle.size
    for spec in input_specs:
        array = bundle.inputs[spec.name]
        if array.ndim != spec.rank:
            raise ValueError(
                f"Input {spec.name!r} has dataset rank {array.ndim}, but model input rank is {spec.rank}. "
                "The first dimension is the dataset batch/sample dimension."
            )
        if array.shape[0] < 1:
            raise ValueError(f"Input {spec.name!r} contains zero samples.")
        for dim_idx, expected in enumerate(spec.shape):
            if dim_idx == 0 or expected is None:
                continue
            actual = int(array.shape[dim_idx])
            if actual != expected:
                raise ValueError(
                    f"Shape mismatch for {spec.name!r}: model expects dimension {dim_idx}={expected}, "
                    f"but dataset provides {actual}."
                )
    return size


def validate_dataset(
    bundle: DatasetBundle,
    input_specs: Sequence[InputSpec],
    require_targets: bool = False,
) -> None:
    """Validate batch counts, ranks, fixed dimensions, and numeric safety."""
    if not bundle.inputs:
        raise ValueError("Dataset contains no input tensors.")

    expected_names = [spec.name for spec in input_specs]
    if list(bundle.inputs.keys()) != expected_names:
        # Order matters for TFLite representative data. Rebuild in model order.
        bundle.inputs = {spec.name: bundle.inputs[spec.name] for spec in input_specs}

    size = _validate_dataset_shapes(bundle, input_specs)

    if bundle.targets is not None and len(bundle.targets) != size:
        raise ValueError(
            f"Target sample count {len(bundle.targets)} does not match input sample count {size}."
        )
    if require_targets and bundle.targets is None:
        raise ValueError("Evaluation requires targets. Put __target__ in the NPZ or pass --labels.")


def representative_generator(
    bundle: DatasetBundle,
    input_specs: Sequence[InputSpec],
    num_samples: int = 100,
) -> Iterable[list[np.ndarray]]:
    """Yield representative samples in model input order."""
    count = min(bundle.size, max(1, int(num_samples)))
    for index in range(count):
        sample: list[np.ndarray] = []
        for spec in input_specs:
            value = bundle.inputs[spec.name][index : index + 1]
            try:
                cast_value = value.astype(spec.dtype, copy=False)
            except TypeError as exc:
                raise ValueError(
                    f"Could not cast calibration input {spec.name!r} to {spec.dtype}."
                ) from exc
            sample.append(cast_value)
        yield sample


def validate_int8_calibration(bundle: DatasetBundle, input_specs: Sequence[InputSpec]) -> None:
    """Require numeric tensors suitable for full integer quantization."""
    for spec in input_specs:
        if not np.issubdtype(bundle.inputs[spec.name].dtype, np.number):
            raise ValueError(
                f"INT8 PTQ requires numeric calibration tensors. Input {spec.name!r} has dtype "
                f"{bundle.inputs[spec.name].dtype}."
            )


def _resolve_shape_for_input(
    spec: InputSpec,
    data_shape: tuple[int, ...] | None,
    override: tuple[int, ...] | None,
) -> tuple[int | None, ...]:
    shape = list(spec.shape)
    if override is not None and len(override) != len(shape):
        raise ValueError(
            f"Explicit shape for {spec.name!r} has rank {len(override)}, expected {len(shape)}."
        )

    for dim_idx, dim in enumerate(shape):
        if dim_idx == 0 and dim is None:
            shape[dim_idx] = 1
            continue

        if dim is not None:
            if (
                override is not None
                and dim_idx < len(override)
                and int(override[dim_idx]) != int(dim)
            ):
                raise ValueError(
                    f"Explicit shape for {spec.name!r} conflicts with fixed model dimension "
                    f"{dim_idx}: model={dim}, override={override[dim_idx]}."
                )
            continue

        candidate: int | None = None
        if override is not None and dim_idx < len(override):
            candidate = int(override[dim_idx])
        elif data_shape is not None:
            candidate = int(data_shape[dim_idx])

        if candidate is None or candidate <= 0:
            raise ValueError(
                f"Input {spec.name!r} has unresolved dynamic dimension {dim_idx}. "
                "Provide calibration/evaluation data or --input-shape for this input."
            )
        shape[dim_idx] = candidate

    return tuple(shape)


def resolve_dynamic_shapes(
    input_specs: Sequence[InputSpec],
    bundle: DatasetBundle | None = None,
    explicit_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> list[InputSpec]:
    """Resolve non-batch dynamic dimensions from data or explicit CLI shapes.

    The function never invents 128/224/etc. If a required dimension is unknown
    after checking data and explicit overrides, it raises an actionable error.
    """
    explicit_shapes = explicit_shapes or {}
    resolved: list[InputSpec] = []
    for spec in input_specs:
        data_shape = (
            bundle.inputs[spec.name].shape
            if bundle is not None and spec.name in bundle.inputs
            else None
        )
        override = explicit_shapes.get(spec.name)
        if override is None:
            override = explicit_shapes.get(canonical_name(spec.name))

        resolved.append(
            InputSpec(spec.name, _resolve_shape_for_input(spec, data_shape, override), spec.dtype)
        )
    return resolved
