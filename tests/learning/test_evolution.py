"""Phase-8 continuous-evolution tests (docs/ROADMAP.md § Phase 8)."""

from __future__ import annotations

from astra.learning.evolution import (
    DriftDetector,
    MetricsStore,
    PromotionSnapshot,
    lineage,
)


def _rows(values: list[float], metric: str = "regression_astra_loss") -> list[dict]:
    out = []
    for i, v in enumerate(values):
        out.append({
            "name": "test-model",
            "sha256": f"{i:064x}",
            "semver": f"1.{i}.0",
            "metrics": {metric: v},
            "created_at": "2026-09-11T00:00:00Z",
            "drift": {},
        })
    return out


def test_metrics_store_roundtrip(tmp_path):
    store = MetricsStore(root=str(tmp_path / "m"))
    store.record(PromotionSnapshot(name="n", sha256="aa", semver="1.0.0",
                                   metrics={"a": 1.0, "b": 2.0}))
    store.record(PromotionSnapshot(name="n", sha256="bb", semver="1.1.0",
                                   metrics={"a": 0.5}, report_id="r2"))
    series = store.series("n")
    assert len(series) == 2
    assert series[0]["sha256"] == "aa"
    assert series[1]["metrics"] == {"a": 0.5}
    assert store.last("n")["sha256"] == "bb"
    assert store.metric_names("n") == ["a", "b"]
    assert store.series("other") == []
    assert store.last("other") is None


def test_drift_needs_more_data():
    det = DriftDetector()
    assert det.check(_rows([1.0, 0.9]), "a").trend == "needs_more_data"


def test_drift_improving_series_is_clean():
    det = DriftDetector(regression_threshold=0.1, drift_tolerance=0.2)
    r = det.check(_rows([2.0, 1.5, 1.2, 1.0, 0.8]), "regression_astra_loss")
    assert r.trend == "improving"
    assert r.drift is False


def test_drift_stable_series_is_clean():
    det = DriftDetector(regression_threshold=0.1, drift_tolerance=0.2)
    r = det.check(_rows([1.0, 1.02, 0.99, 1.01, 1.0]), "regression_astra_loss")
    assert r.trend == "stable"
    assert r.drift is False


def test_drift_regressing_series_fires():
    det = DriftDetector(regression_threshold=0.1, drift_tolerance=0.2)
    r = det.check(_rows([0.8, 1.0, 1.4, 1.9, 2.5]), "regression_astra_loss")
    assert r.trend == "regressing"
    assert r.drift is True


def test_drift_silent_crawl_fires_drift_not_regress():
    # each step is under the regression threshold, but the cumulative move
    # from the series best exceeds the drift tolerance -> silent drift.
    det = DriftDetector(regression_threshold=0.1, drift_tolerance=0.1)
    vals = [1.0, 1.04, 1.08, 1.12, 1.16, 1.20, 1.24, 1.28]
    r = det.check(_rows(vals), "regression_astra_loss")
    assert r.trend == "stable"
    assert r.drift is True
    assert "silent drift" in r.message


def test_lineage_from_registry(tmp_path):
    from astra.registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path / "r.json"))
    for sv, step in (("1.0.0", 10), ("1.1.0", 20), ("1.2.0", 30)):
        ckpt = tmp_path / f"{sv}.npz"
        ckpt.write_bytes(f"v{sv}".encode())
        rec = reg.register(name="m", semver=sv, path=str(ckpt),
                           config_id="c", data_manifest_id="d",
                           checklist_id="phase8-test", params=6, step=step)
        reg.promote("m", rec.sha256, notes="chain")

    chain = lineage("m", reg)
    assert [c["semver"] for c in chain] == ["1.0.0", "1.1.0", "1.2.0"]
    assert chain[0]["superseded_by"] == chain[1]["sha256"]
    assert chain[1]["superseded_by"] == chain[2]["sha256"]
    assert chain[-1]["superseded_by"] == ""
    assert [c["step"] for c in chain] == [10, 20, 30]