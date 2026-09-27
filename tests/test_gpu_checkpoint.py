"""Regression: a bf16 GPU training run must be able to save a checkpoint.

`save_checkpoint_torch` used `p.detach().cpu().numpy()`. numpy has no bfloat16, so that
raises `TypeError: Got unsupported ScalarType BFloat16` on a bf16 run. Because the first
snapshot is taken right after step 0, the run died having learned nothing - and the
module docstring's promise that the .npz "loads unchanged in the NumPy inference path"
was false, since NumPy has no bf16 to load.

These tests skip without torch, so they are collected here but only enforced on a GPU host.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed - GPU checkpoint test skipped")

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "gpu_train", Path(__file__).resolve().parents[1] / "training" / "gpu_train.py"
)
gpu_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu_train)


BF16_DTYPES = [torch.bfloat16]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    BF16_DTYPES.append(torch.float16)


@pytest.mark.parametrize("dtype", BF16_DTYPES)
def test_to_numpy_downcasts_bfloat16_to_float32(dtype):
    """The whole point: numpy must be able to materialise the saved tensor."""
    t = torch.ones(4, 5, dtype=dtype)
    arr = gpu_train._to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    np.savez_compressed("/tmp/_astra_bf16_probe.npz", w=arr)  # the failing operation
    assert np.load("/tmp/_astra_bf16_probe.npz")["w"].shape == (4, 5)


def test_to_numpy_preserves_values():
    t = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float32)
    np.testing.assert_allclose(gpu_train._to_numpy(t), t.numpy())


def test_to_numpy_accepts_a_grad_tracking_tensor():
    """A bare `.numpy()` raises on a tensor that requires grad. The cast is done via
    `.detach().to(...)`, so this must not - and it must not silently return a graph."""
    t = torch.ones(3, requires_grad=True)
    arr = gpu_train._to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32

    with pytest.raises(RuntimeError):
        t.numpy()


def test_numpy_still_cannot_read_bfloat16():
    """Guards the premise. If a future numpy adds bf16, this fails loudly and the cast in
    _to_numpy can be revisited rather than left as an unexplained widening."""
    with pytest.raises(TypeError):
        np.dtype("bfloat16")
