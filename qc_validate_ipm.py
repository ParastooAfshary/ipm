#!/usr/bin/env python3
r"""Independent numerical QC of existing single-subject IPM FC outputs.

Run from the ipm-1sub folder after run_fc.py has completed:
    .venv\Scripts\python.exe qc_validate_ipm.py

This script does NOT modify raw MRI files or rerun preprocessing.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    out = Path(__file__).resolve().parent / "results_subj01"
    needed = [
        "pearson_fc.npy", "fisher_z_fc.npy", "roi_timeseries.npy",
        "fisher_z_edges.npy", "roi_metadata.csv", "qc_summary.json",
    ]
    missing = [name for name in needed if not (out / name).is_file()]
    if missing:
        print("ERROR: Missing output files:", ", ".join(missing))
        print("Put qc_validate_ipm.py inside the ipm-1sub folder, next to run_fc.py.")
        return 1

    fc = np.load(out / "pearson_fc.npy", allow_pickle=False)
    z = np.load(out / "fisher_z_fc.npy", allow_pickle=False)
    ts = np.load(out / "roi_timeseries.npy", allow_pickle=False)
    edges = np.load(out / "fisher_z_edges.npy", allow_pickle=False)
    with (out / "qc_summary.json").open(encoding="utf-8") as f:
        original_qc = json.load(f)
    with (out / "roi_metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        roi_names = [row["roi"] for row in csv.DictReader(f)]

    checks = {}
    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)
        print(f"  {'PASS' if condition else 'FAIL'}  {name}")

    print("\n=== IPM independent FC quality check ===")
    if ts.ndim != 2 or fc.ndim != 2 or z.ndim != 2:
        print(f"ERROR: Unexpected array dimensions: ts={ts.shape}, fc={fc.shape}, z={z.shape}")
        return 1
    frames, rois = ts.shape
    expected_edges = rois * (rois - 1) // 2
    check("Shapes consistent", fc.shape == (rois, rois) and z.shape == fc.shape)
    check("ROI metadata count", len(roi_names) == rois)
    check("Frame count matches original QC", original_qc.get("frames_retained") == frames)
    check("ROI count matches original QC", original_qc.get("n_rois") == rois)
    check("No NaN/Inf in ROI series or matrices", bool(np.isfinite(ts).all() and np.isfinite(fc).all() and np.isfinite(z).all()))
    check("Pearson correlations within [-1, 1]", bool(np.max(np.abs(fc)) <= 1.00000001))
    check("Pearson matrix symmetric", bool(np.allclose(fc, fc.T, atol=1e-10, rtol=0)))
    check("Pearson diagonal equals 1", bool(np.allclose(np.diag(fc), 1.0, atol=1e-10, rtol=0)))
    check("Fisher matrix symmetric", bool(np.allclose(z, z.T, atol=1e-10, rtol=0)))
    check("Fisher diagonal equals 0 (self-edges excluded)", bool(np.allclose(np.diag(z), 0.0, atol=1e-10, rtol=0)))
    check("Edge vector length", edges.shape == (expected_edges,))

    if rois >= 2 and np.isfinite(fc).all() and np.isfinite(z).all():
        upper = np.triu_indices(rois, 1)
        rvals = fc[upper]
        expected_z = np.arctanh(np.clip(rvals, -1.0 + 1e-7, 1.0 - 1e-7))
        check("Fisher z matches Pearson r", bool(np.allclose(z[upper], expected_z, atol=1e-8, rtol=1e-8)))
        check("Saved edge vector matches Fisher matrix", bool(np.allclose(edges, z[upper], atol=1e-10, rtol=0)))
        if np.isfinite(ts).all() and frames >= 3:
            recalc = np.corrcoef(ts, rowvar=False)
            delta = float(np.max(np.abs(recalc - fc))) if np.isfinite(recalc).all() else None
            check("Pearson FC reproducible from ROI series", delta is not None and delta < 1e-7)
        else:
            delta = None
            check("Pearson FC reproducible from ROI series", False)
        stats = {
            "pearson_mean": float(np.mean(rvals)),
            "pearson_median": float(np.median(rvals)),
            "pearson_min": float(np.min(rvals)),
            "pearson_max": float(np.max(rvals)),
            "pearson_p05": float(np.percentile(rvals, 5)),
            "pearson_p95": float(np.percentile(rvals, 95)),
            "fraction_abs_r_over_0_9": float(np.mean(np.abs(rvals) > 0.9)),
            "fraction_abs_r_over_0_99": float(np.mean(np.abs(rvals) > 0.99)),
        }
    else:
        stats = {}
        delta = None
        rvals = np.array([])

    excluded = original_qc.get("surface_vertices", {})
    summary = {
        "subject": original_qc.get("subject", "unknown"),
        "atlas": original_qc.get("atlas", "unknown"),
        "frames_retained": frames,
        "frames_total": original_qc.get("frames_total"),
        "n_rois": rois,
        "n_unique_edges": expected_edges,
        "max_abs_difference_recomputed_fc": delta,
        "vertex_qc": excluded,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "fc_statistics": stats,
        "interpretation_limitations": [
            "Passing numerical checks does not establish scientific validity or individual-network reproducibility.",
            "Censor file polarity and surface-to-atlas vertex correspondence should be confirmed against source documentation.",
            "Destrieux is an anatomical atlas, not Kong MS-HBM individual-specific network parcellation.",
            "Single session/subject cannot establish test-retest stability or between-subject effects.",
        ],
    }
    (out / "qc_validation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nData: {frames} frames x {rois} ROIs | {expected_edges} unique edges")
    if stats:
        print(f"Off-diagonal Pearson r: min={stats['pearson_min']:.3f}, median={stats['pearson_median']:.3f}, max={stats['pearson_max']:.3f}")
        print(f"High absolute correlations: |r|>0.90: {100*stats['fraction_abs_r_over_0_9']:.2f}% | |r|>0.99: {100*stats['fraction_abs_r_over_0_99']:.2f}%")
    if rvals.size:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(rvals, bins=60, edgecolor="none")
        ax.set(xlabel="Pearson r (unique ROI pairs)", ylabel="Number of edges", title="Subject 01 | Distribution of ROI-to-ROI functional connectivity")
        fig.tight_layout()
        fig.savefig(out / "pearson_edge_distribution.png", dpi=180)
        plt.close(fig)
    print(f"\nRESULT: {'ALL NUMERICAL CHECKS PASSED' if summary['all_checks_passed'] else 'ONE OR MORE CHECKS FAILED'}")
    print(f"Saved: {out / 'qc_validation.json'}")
    if rvals.size:
        print(f"Saved: {out / 'pearson_edge_distribution.png'}")
    print("Note: Numerical QC is not a substitute for visual inspection and scientific validation.")
    return 0 if summary["all_checks_passed"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
