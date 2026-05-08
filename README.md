# sparse_mla_sm120

CUDA kernel library for **DeepSeek-style sparse attention**, providing:

- **Sparse MLA**: **Prefill** and **decode** forwards with FP8-packed KV and top‑k indices (`sparse_mla_prefill_fwd`, `sparse_mla_decode_fwd`, etc.; see `sparse_mla_sm120/ops.py`).

## Target hardware

- **Architecture**: NVIDIA **compute capability 12.x (SM120 family)**. The build emits both **`sm_120a`** and **`sm_120f`** (see `-gencode` flags in `setup.py`).

## Requirements

- **Python** ≥ 3.10 (matches `python_requires` in `setup.py`)
- **PyTorch with CUDA** (the extension is built via `torch.utils.cpp_extension`)
- **CUDA Toolkit / nvcc** compatible with your GPU and able to compile for SM120

## Installation

From the repository root:

```bash
python3 setup.py bdist_wheel
```

The wheel is written under `dist/`. Install it with:

```bash
pip install dist/sparse_mla_sm120-*.whl
```

For development, an editable install is also fine:

```bash
pip install -e .
```

## Benchmarks

Run from the repository root (after installing the package or adding the root to `PYTHONPATH`).

| Script | Purpose |
|--------|---------|
| `benchmarks/benchmark_sparse_mla.py` | Sparse MLA **prefill / decode**: latency, effective bandwidth, TFLOP/s (DeepSeek V3.2–scale settings) |
| `scripts/bench.py` | Simple **prefill** benchmark (uses helpers under `tests/`; run from repo root) |

Examples:

```bash
python benchmarks/benchmark_sparse_mla.py
```

## Tests

With **pytest** installed:

```bash
pytest tests/test_sparse_mla.py -v -s
```

Tests compare CUDA outputs to PyTorch references; tolerances account for FP8 quantization.
