# sparse_mla_sm120

CUDA kernel library for **DeepSeek-style sparse attention**, providing:

- **Sparse MLA**: **Prefill** and **decode** forwards with FP8-packed KV and top‑k indices (`sparse_mla_prefill_fwd`, `sparse_mla_decode_fwd`, etc.; see `sparse_mla_sm120/ops.py`).
- **MQA logits**: **MQA logits** from FP8 Q and KV.

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
| `benchmarks/benchmark_mqa_logits.py` | FP8 **MQA logits**: CUDA vs Triton; `--mode ragged \| paged \| all` (default `all`) |
| `benchmarks/compare_deepseek_mla_b12x.py` | Compare this repo’s decode vs **b12x** sparse MLA ([b12x](https://github.com/lukealonso/b12x) must be installed separately; set `B12X_ROOT` or `pip install -e`) |
| `scripts/bench.py` | Simple **prefill** benchmark (uses helpers under `tests/`; run from repo root) |

Examples:

```bash
python benchmarks/benchmark_sparse_mla.py
python benchmarks/benchmark_mqa_logits.py --mode all
python benchmarks/benchmark_mqa_logits.py --mode ragged
```

## Tests

With **pytest** installed:

```bash
pytest tests/test_sparse_mla.py -v -s
pytest tests/test_mqa_logits.py -v -s
```

Or all tests under `tests/`:

```bash
pytest tests/ -v
```

Tests compare CUDA outputs to PyTorch references; sparse MLA uses relaxed tolerances due to FP8 quantization.
