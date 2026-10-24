"""Reporting must distinguish unfinished experiments from failed SLOs."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.make_report import build_summary
from tests.test_aggregate import point


def test_report_reads_serial_overhead_without_openloop_schema(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "overhead"
    raw.mkdir(parents=True)
    rows: list[dict[str, Any]] = [{"meta": {"kind": "serial_ab"}}]
    for target in ("direct", "gateway"):
        rows.append({"pair": 0, "target": target, "status": 200, "error": None, "ttft_s": 0.2, "e2e_s": 1.0})
    (raw / "serial_ab.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    summary = build_summary(tmp_path)
    assert summary["overhead"]["serial"]["n_pairs"] == 1
    assert summary["overhead"]["openloop"] == []


def test_partial_grid_cannot_claim_capacity(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "capacity"
    raw.mkdir(parents=True)
    p = point("Q4_K_M", 0.3, [0.1] * 20)
    (raw / "point.jsonl").write_text(
        json.dumps({"meta": p.meta}) + "\n" + "\n".join(json.dumps(asdict(r)) for r in p.results)
    )
    summary = build_summary(tmp_path)
    assert len(summary["capacity"]) == 6
    assert all(c["capacity_rps"] is None and c["bound"] == "incomplete" for c in summary["capacity"])


def test_report_uses_recorded_grid_with_json_string_keys(tmp_path: Path) -> None:
    cfg = {
        "sweep": {"model": "m", "quants": ["Q"], "parallel": [1], "rates": {"1": [0.1]}, "repeats": 1},
        "slo": {"chat": {"ttft_p95_s": 0.5}},
        "models": {"m": {"reference_quant": "Q"}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"configs": cfg}))
    summary = build_summary(tmp_path)
    assert summary["model"] == "m"
    assert summary["slo_chat_ttft_p95_s"] == 0.5
    assert len(summary["capacity"]) == 1
    assert summary["capacity"][0]["missing_points"] == [{"rate": 0.1, "rep": 0}]


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    import pytest

    from scripts.run_sweep import load_configs, update_manifest

    cfg = load_configs()
    (tmp_path / "manifest.json").write_text(json.dumps({"configs": cfg}))
    cfg["sweep"]["duration_s"] = 1
    with pytest.raises(ValueError, match="configuration"):
        update_manifest(tmp_path, cfg, "capacity", "started")


def test_report_fills_initial_empty_markers(tmp_path: Path) -> None:
    from scripts.make_report import inject

    doc = tmp_path / "README.md"
    doc.write_text("intro\n<!-- BEGIN:table1 -->\n<!-- END:table1 -->\nend\n")
    inject(doc, {"table1": "generated table"})
    assert "<!-- BEGIN:table1 -->\ngenerated table\n<!-- END:table1 -->" in doc.read_text()
    inject(doc, {"table1": "updated"})
    assert "generated table" not in doc.read_text()
    assert "\nupdated\n" in doc.read_text()
