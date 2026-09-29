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


# --------------------------------------------------------------- resume points
# A 50k-step run is 10-25 GPU hours and Colab reclaims preemptible runtimes without warning.
# The only state that can continue a run is a full one: weights, the AdamW moments, and the
# step counter. The cheap weights-only snapshots cannot, and gpu_train.py refuses them rather
# than restart AdamW against a nonzero step counter and train on a lie. These tests pin that
# division down, because it is invisible until a runtime dies and the only test is whether
# the run continues.

_TINY_MODEL = {
    "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2, "d_head": 16,
    "d_ffn": 64, "max_seq_len": 32, "rope_theta": 10000.0, "eps": 1e-6,
    "tie_embeddings": True,
}


def _tiny_pair(tmp_path, seed=0):
    torch.manual_seed(seed)
    model = gpu_train.TorchLiteLM(gpu_train.ModelConfig(**_TINY_MODEL), seed=seed,
                                  device=torch.device("cpu"), dtype=torch.float32)
    opt = gpu_train.AdamW(model, lr=1e-3, weight_decay=0.05)
    return model, opt


def _train_a_few_steps(model, opt, n=5):
    """Move the weights and the AdamW moments off their initial values."""
    for i in range(n):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step(lr=1e-3 * (i + 1))
        for p in model.parameters():
            p.grad = None


def test_resume_point_restores_weights_adam_state_and_step(tmp_path):
    """The whole feature. If any of the three came back wrong the run would either lose its
    optimizer or silently restart the schedule."""
    model, opt = _tiny_pair(tmp_path)
    _train_a_few_steps(model, opt)
    path = str(tmp_path / "resume-5.npz")
    gpu_train.save_checkpoint_torch(
        path, model, opt, None, step=5, loss_hist=[1.0, 2.0],
        meta={"kind": "resume_point"}, include_optimizer=True, compress=False)

    fresh_model, fresh_opt = _tiny_pair(tmp_path, seed=999)  # deliberately different init
    step, hist = gpu_train.load_checkpoint_torch(path, fresh_model, fresh_opt)

    assert step == 5, 'the step counter is what places the LR on the schedule'
    assert hist == [1.0, 2.0]
    for (n, p), (n2, q) in zip(model.named_parameters(), fresh_model.named_parameters()):
        assert n == n2
        torch.testing.assert_close(p, q, rtol=0, atol=1e-6, msg=f"weight {n} not restored")
    saved = np.load(path)
    for name, arr in opt.state_dict().items():
        torch.testing.assert_close(arr, torch.from_numpy(saved[name]),
                                   rtol=0, atol=1e-6, msg=f"adam state {name} not restored")


def test_weights_only_snapshot_is_refused_rather_than_half_resumed(tmp_path):
    """The trap this design exists to avoid: a 3.3 GB snapshot that looks like a checkpoint
    but would restart AdamW's bias correction at a nonzero step."""
    model, opt = _tiny_pair(tmp_path)
    path = str(tmp_path / "checkpoint-5.npz")
    gpu_train.save_checkpoint_torch(path, model, opt, None, step=5, loss_hist=[],
                                    meta={"kind": "intermediate"}, include_optimizer=False)
    _model2, opt2 = _tiny_pair(tmp_path, seed=7)
    with pytest.raises(ValueError, match="weights-only"):
        gpu_train.load_checkpoint_torch(path, _model2, opt2)


def test_compression_does_not_change_what_loads(tmp_path):
    """A resume point is written uncompressed for speed, so the loader has to treat both
    containers the same or resuming depends on how a file happened to be written."""
    model, opt = _tiny_pair(tmp_path)
    _train_a_few_steps(model, opt, n=2)
    results = []
    for compress in (True, False):
        path = str(tmp_path / (("c" if compress else "u") + ".npz"))
        gpu_train.save_checkpoint_torch(path, model, opt, None, step=2, loss_hist=[],
                                        meta={}, include_optimizer=True, compress=compress)
        m2, o2 = _tiny_pair(tmp_path, seed=31)
        step, _ = gpu_train.load_checkpoint_torch(path, m2, o2)
        results.append((step, [p.detach().clone() for p in m2.parameters()]))
    assert results[0][0] == results[1][0] == 2
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_manifest_records_compression_and_optimizer_state(tmp_path):
    """find_resume_point() in the notebook trusts the manifest to tell a resumable file from
    a cheap one, so the flags it reads have to actually be written."""
    model, opt = _tiny_pair(tmp_path)
    import json as _json
    for compress, include in ((False, True), (True, False)):
        path = str(tmp_path / "resume-1.npz")
        gpu_train.save_checkpoint_torch(path, model, opt, None, step=1, loss_hist=[],
                                        meta={}, include_optimizer=include, compress=compress)
        man = _json.loads((tmp_path / "resume-1.manifest.json").read_text())
        assert man["optimizer_included"] is include
        assert man["compressed"] is compress
        assert man["step"] == 1
        assert man["weights_dtype"] == "float16"


def test_reset_step_warm_start_keeps_weights_but_restarts_the_schedule(tmp_path):
    """The chat stage warm-starts from the alerts final with --reset-step, so this is the
    path that has to work, and it must not inherit the alerts run's step counter."""
    model, opt = _tiny_pair(tmp_path)
    _train_a_few_steps(model, opt, n=4)
    path = str(tmp_path / "final.npz")
    gpu_train.save_checkpoint_torch(path, model, opt, None, step=4, loss_hist=[1.0],
                                    meta={}, include_optimizer=True)

    m2, o2 = _tiny_pair(tmp_path, seed=123)
    step, hist = gpu_train.load_checkpoint_torch(path, m2, o2, reset_step=True)
    assert (step, hist) == (0, []), 'a warm start must begin a new schedule'
    original = dict(model.named_parameters())
    for name, p in m2.named_parameters():
        torch.testing.assert_close(original[name], p, rtol=0, atol=1e-6,
                                   msg=f"weight {name} did not come across")

