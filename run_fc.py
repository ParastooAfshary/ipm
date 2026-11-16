#!/usr/bin/env python3
"""Single-subject surface-based resting-state FC analysis for CoRR-HNU CBIG example.

Inputs: preprocessed lh/rh fsaverage5 NIfTI (.nii.gz), CBIG 0/1 censor file.
Atlas: Destrieux 2010 cortical surface atlas on fsaverage5 (fixed anatomical atlas).
This is NOT the Kong MS-HBM individual parcellation or Ladwig LPFC replication.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def choose_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {folder}; "
            f"found {len(matches)}: {[p.name for p in matches]}"
        )
    return matches[0]


def locate_inputs(folder: Path) -> dict[str, Path]:
    return {
        "lh_fs5": choose_one(folder, "lh.*_fs6_sm6_fs5.nii.gz"),
        "rh_fs5": choose_one(folder, "rh.*_fs6_sm6_fs5.nii.gz"),
        "censor": choose_one(folder, "*motion_outliers.txt"),
    }


def reshape_surface_vertices_time(data: np.ndarray, n_vertices: int = 10242) -> np.ndarray:
    """Read CBIG surface NIfTI into (fsaverage5 vertex, time).

    CBIG can pack the 10,242 fsaverage5 vertices into spatial dimensions
    (3414, 1, 3) rather than (10242, 1, 1).  MATLAB/FreeSurfer surface
    indexing uses the first spatial axis fastest, so use Fortran-order
    flattening of the spatial axes, NEVER a default C-order reshape.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        if data.shape[0] == n_vertices:
            return data
        if data.shape[1] == n_vertices:
            return data.T
    elif data.ndim in (3, 4):
        n_spatial = int(np.prod(data.shape[:-1]))
        if n_spatial == n_vertices:
            return np.reshape(data, (n_vertices, data.shape[-1]), order="F")
    raise ValueError(
        f"Cannot interpret surface NIfTI shape {data.shape}: expected "
        f"{n_vertices} vertices in the spatial axes and time in the last axis."
    )


def read_surface_nifti(path: Path, n_vertices: int = 10242) -> tuple[np.ndarray, tuple[int, ...]]:
    import nibabel as nib

    img = nib.load(str(path))
    original_shape = tuple(int(x) for x in img.shape)
    try:
        data = reshape_surface_vertices_time(img.get_fdata(dtype=np.float32), n_vertices)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc} (NIfTI shape: {original_shape})") from exc
    if data.shape[1] < 30:
        raise ValueError(f"{path.name}: only {data.shape[1]} time frames found; please check the file")
    # Surface MRI can contain NaN/Inf at medial-wall or otherwise invalid vertices.
    # Keep the raw data unchanged; ROI extraction will exclude entire invalid vertices.
    nonfinite_vertices = int((~np.isfinite(data).all(axis=1)).sum())
    if nonfinite_vertices:
        print(
            f"      QC {path.name[:2]}: {nonfinite_vertices}/{data.shape[0]} vertices "
            "contain NaN/Inf and will be excluded from ROI averaging.",
            flush=True,
        )
    return data, original_shape


def read_censor(path: Path, n_frames: int) -> tuple[np.ndarray, str]:
    """CBIG documentation: 0 denotes censored/outlier frame; 1 is kept."""
    try:
        raw = np.loadtxt(path, dtype=float)
    except ValueError as exc:
        raise ValueError(f"Cannot parse censor file {path.name}: {exc}") from exc
    raw = np.asarray(raw).reshape(-1)
    note = "Censor vector matched fMRI length."
    if len(raw) == n_frames + 4:
        raw = raw[4:]
        note = "Censor vector had 4 extra initial frames; trimmed first 4 to match skip4 fMRI."
    if len(raw) != n_frames:
        raise ValueError(
            f"Censor file has {len(raw)} values, fMRI has {n_frames} frames. "
            "Please send this error and the first lines of the censor file before proceeding."
        )
    if not np.all(np.isin(raw, [0, 1])):
        raise ValueError(f"Censor values must be 0/1, got {np.unique(raw)[:20]}")
    keep = raw.astype(bool)
    if np.count_nonzero(keep) < 30:
        raise ValueError(f"Only {np.count_nonzero(keep)} frames remain after censoring; cannot compute reliable FC.")
    return keep, note


def normalize_atlas_labels(raw_labels) -> list[str]:
    out = []
    for item in raw_labels:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="replace"))
        else:
            out.append(str(item))
    return out


def extract_roi_timeseries(
    hemispheres: dict[str, np.ndarray],
    atlas_maps: dict[str, np.ndarray],
    atlas_labels: list[str],
    keep: np.ndarray,
    min_vertices: int = 10,
) -> tuple[np.ndarray, list[dict], dict]:
    columns = []
    metadata = []
    vertex_counts = {}
    for hemi in ("lh", "rh"):
        data = hemispheres[hemi]
        labels = np.asarray(atlas_maps[hemi]).ravel()
        if len(labels) != data.shape[0]:
            raise ValueError(f"Atlas/data mismatch for {hemi}: atlas {len(labels)} vertices vs fMRI {data.shape[0]}")
        # Exclude vertices with any NaN/Inf, and vertices with constant time series.
        # Compute std ONLY on finite vertices to avoid NaN/Inf warnings and contamination.
        finite = np.isfinite(data).all(axis=1)
        varying = np.zeros(data.shape[0], dtype=bool)
        if finite.any():
            varying[finite] = np.std(data[finite], axis=1, dtype=np.float64) > 1e-8
        usable = finite & varying
        vertex_counts[hemi] = {
            "total": int(len(usable)),
            "usable": int(usable.sum()),
            "nonfinite_excluded": int((~finite).sum()),
            "constant_excluded": int((finite & ~varying).sum()),
        }
        for idx in sorted(int(i) for i in np.unique(labels) if int(i) > 0):
            selected = (labels == idx) & usable
            count = int(selected.sum())
            if count < min_vertices:
                continue
            roi = data[selected][:, keep].mean(axis=0, dtype=np.float64)
            if not np.isfinite(roi).all() or np.std(roi) <= 1e-8:
                continue
            region_name = atlas_labels[idx] if idx < len(atlas_labels) else f"label_{idx}"
            safe_name = f"{hemi}_{idx:02d}_{region_name}".replace(" ", "_")
            columns.append(roi)
            metadata.append({
                "roi": safe_name,
                "hemisphere": hemi,
                "atlas_index": idx,
                "atlas_label": region_name,
                "usable_vertices": count,
            })
    if len(columns) < 3:
        raise ValueError(f"Only {len(columns)} usable ROIs found; cannot calculate a meaningful FC matrix.")
    return np.column_stack(columns), metadata, vertex_counts


def compute_fc(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ts.ndim != 2 or ts.shape[0] < 30 or ts.shape[1] < 3:
        raise ValueError(f"Unexpected ROI time series shape: {ts.shape}")
    if not np.isfinite(ts).all():
        raise ValueError("ROI time series contains NaN/Inf")
    fc = np.corrcoef(ts, rowvar=False)
    if not np.isfinite(fc).all():
        raise ValueError("FC contains NaN/Inf, perhaps due to constant ROIs")
    fc = np.clip((fc + fc.T) / 2, -1.0, 1.0)
    np.fill_diagonal(fc, 1.0)
    fisher = np.arctanh(np.clip(fc, -1.0 + 1e-7, 1.0 - 1e-7))
    np.fill_diagonal(fisher, 0.0)  # self-correlations are excluded from feature analysis
    iu = np.triu_indices(fc.shape[0], k=1)
    return fc, fisher, fisher[iu]


def write_matrix_csv(path: Path, matrix: np.ndarray, names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ROI"] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + [f"{float(v):.8g}" for v in row])


def plot_heatmap(matrix: np.ndarray, title: str, path: Path, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    ax.set(title=title, xlabel="ROI index (Destrieux atlas)", ylabel="ROI index (Destrieux atlas)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_timeseries(ts: np.ndarray, names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx in range(min(5, ts.shape[1])):
        col = ts[:, idx]
        z = (col - col.mean()) / max(col.std(), 1e-8)
        ax.plot(z + 4 * idx, linewidth=0.9, label=names[idx][:45])
    ax.set(title="Five ROI time series (z-scored, vertically offset)", xlabel="Retained frame index", ylabel="Standardized signal + offset")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_mshbm_template(outdir: Path, folder: Path, censor: Path) -> None:
    """Generate a template, NOT an executed MATLAB result."""
    left = next(iter(sorted(folder.glob("lh.*_fs6_sm6.nii.gz"))), None)
    right = next(iter(sorted(folder.glob("rh.*_fs6_sm6.nii.gz"))), None)
    if left is None or right is None:
        return
    def matpath(p: Path) -> str:
        return str(p.resolve()).replace("\\", "/").replace("'", "''")
    script = f"""% TEMPLATE ONLY. NOT RUN. Requires MATLAB, CBIG, FreeSurfer dependencies and group priors.
% Source: CBIG Kong2019_MSHBM/examples/README.md (single-subject section)
% Set up CBIG_CODE_DIR and add CBIG paths before running.
params = struct();
params.project_dir = '{matpath(outdir / 'mshbm_output')}';
params.censor_list = {{'{matpath(censor)}'}};
params.lh_fMRI_list = {{'{matpath(left)}'}};
params.rh_fMRI_list = {{'{matpath(right)}'}};
params.target_mesh = 'fsaverage6';
params.w = '50';
params.c = '50';
[lh_labels, rh_labels] = CBIG_MSHBM_parcellation_single_subject(params);
"""
    (outdir / "mshbm_single_subject_TEMPLATE_NOT_RUN.m").write_text(script, encoding="utf-8")


def run(folder: Path, output: Path) -> None:
    import nilearn
    from nilearn.datasets import fetch_atlas_surf_destrieux
    import nibabel

    inputs = locate_inputs(folder)
    output.mkdir(parents=True, exist_ok=True)
    print("[1/6] Reading subject01 fsaverage5 surface fMRI ...", flush=True)
    lh, lh_shape = read_surface_nifti(inputs["lh_fs5"])
    rh, rh_shape = read_surface_nifti(inputs["rh_fs5"])
    if lh.shape[1] != rh.shape[1]:
        raise ValueError(f"Different number of frames: LH {lh.shape}, RH {rh.shape}")
    print(f"      LH: {lh.shape}; RH: {rh.shape}")
    keep, censor_note = read_censor(inputs["censor"], lh.shape[1])
    print(f"      Retained {int(keep.sum())}/{len(keep)} frames; {censor_note}")

    print("[2/6] Loading Destrieux surface atlas (fsaverage5) ...", flush=True)
    print("      First run may download a small atlas from the internet.", flush=True)
    atlas = fetch_atlas_surf_destrieux()
    atlas_maps = {"lh": np.asarray(atlas["map_left"]), "rh": np.asarray(atlas["map_right"])}
    atlas_labels = normalize_atlas_labels(atlas["labels"])
    ts, metadata, vertex_counts = extract_roi_timeseries(
        {"lh": lh, "rh": rh}, atlas_maps, atlas_labels, keep
    )
    names = [m["roi"] for m in metadata]
    print(f"      Extracted {ts.shape[1]} ROIs x {ts.shape[0]} retained frames")

    print("[3/6] Calculating Pearson FC ...", flush=True)
    fc, fisher, features = compute_fc(ts)
    print("[4/6] Fisher z and upper-triangle features ...", flush=True)
    np.save(output / "pearson_fc.npy", fc)
    np.save(output / "fisher_z_fc.npy", fisher)
    np.save(output / "fisher_z_edges.npy", features)
    np.save(output / "roi_timeseries.npy", ts)
    write_matrix_csv(output / "pearson_fc.csv", fc, names)
    write_matrix_csv(output / "fisher_z_fc.csv", fisher, names)
    with (output / "roi_timeseries.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["original_frame_index"] + names)
        for original_idx, row in zip(np.flatnonzero(keep), ts):
            writer.writerow([int(original_idx)] + [f"{float(v):.8g}" for v in row])
    with (output / "roi_metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    iu = np.triu_indices(len(names), 1)
    with (output / "fc_edges.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["roi_1", "roi_2", "pearson_r", "fisher_z"])
        for i, j in zip(*iu):
            writer.writerow([names[i], names[j], f"{fc[i, j]:.8g}", f"{fisher[i, j]:.8g}"])

    print("[5/6] Saving figures ...", flush=True)
    plot_heatmap(fc, f"Subject 01 | Pearson FC | {len(names)} Destrieux ROIs", output / "pearson_fc_heatmap.png", -1, 1)
    plot_heatmap(fisher, f"Subject 01 | Fisher z FC | {len(names)} Destrieux ROIs (diagonal set to 0)", output / "fisher_z_heatmap.png")
    plot_timeseries(ts, names, output / "roi_timeseries.png")

    print("[6/6] QC report and next-stage manifest ...", flush=True)
    report = {
        "status": "completed_real_data_analysis",
        "subject": "CoRR-HNU subj01 session1",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "atlas": "Destrieux 2010 anatomical cortical surface atlas, fsaverage5",
        "note": "Atlas is fixed anatomical ROIs, NOT subject-specific MS-HBM network parcellation.",
        "input_files": {k: str(v.resolve()) for k, v in inputs.items()},
        "nifti_shapes": {"lh": lh_shape, "rh": rh_shape},
        "surface_vertices": vertex_counts,
        "frames_total": int(len(keep)),
        "frames_retained": int(keep.sum()),
        "frames_censored": int((~keep).sum()),
        "censor_note": censor_note,
        "n_rois": int(len(names)),
        "n_fc_edges": int(len(features)),
        "off_diagonal_mean_pearson": float(np.mean(fc[iu])),
        "off_diagonal_mean_absolute_pearson": float(np.mean(np.abs(fc[iu]))),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "nilearn": nilearn.__version__,
        "nibabel": nibabel.__version__,
        "next_steps": [
            "MS-HBM individual-specific parcellation using official CBIG MATLAB implementation and fsaverage6 files",
            "Ladwig LPFC analyses after checking compatible data and network definitions",
            "T1 + CHARM + SimNIBS E-field simulation on matched subject when structural MRI is available",
            "Cross-subject SNF and clustering only after multiple subjects are processed",
        ],
    }
    (output / "qc_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_mshbm_template(output, folder, inputs["censor"])
    print(f"\nSUCCESS! Outputs saved in: {output.resolve()}", flush=True)
    print("Open pearson_fc_heatmap.png and qc_summary.json to inspect results.", flush=True)


def self_test() -> None:
    rng = np.random.default_rng(123)
    ts = rng.normal(size=(100, 6))
    ts[:, 1] = ts[:, 0] * 0.8 + rng.normal(size=100) * 0.2
    fc, z, vec = compute_fc(ts)
    assert fc.shape == (6, 6) and z.shape == (6, 6) and vec.shape == (15,)
    assert np.allclose(fc, fc.T) and np.allclose(np.diag(fc), 1)
    assert np.allclose(np.diag(z), 0)
    assert fc[0, 1] > 0.9
    # Reproduce the observed CBIG fsaverage5 storage shape (3414,1,3,T).
    # Check that the vertex order survives spatial unpacking.
    expected = np.arange(10242 * 8, dtype=np.float32).reshape(10242, 8, order="F")
    packed = expected.reshape(3414, 1, 3, 8, order="F")
    unpacked = reshape_surface_vertices_time(packed)
    assert unpacked.shape == (10242, 8) and np.array_equal(unpacked, expected)
    assert np.array_equal(reshape_surface_vertices_time(expected), expected)
    # Nonfinite/constant vertices should be excluded, without altering other time series.
    toy = rng.normal(size=(24, 60)).astype(np.float32)
    toy[0, 0] = np.nan
    toy[8, 1] = np.inf
    toy[16, :] = 0.0
    labels = np.repeat([1, 2, 3], 8)
    roi_ts, meta, qc = extract_roi_timeseries(
        {"lh": toy, "rh": toy.copy()},
        {"lh": labels, "rh": labels},
        ["background", "region1", "region2", "region3"],
        np.ones(60, dtype=bool),
        min_vertices=3,
    )
    assert roi_ts.shape == (60, 6) and np.isfinite(roi_ts).all()
    assert qc["lh"]["nonfinite_excluded"] == 2
    assert qc["lh"]["constant_excluded"] == 1
    assert len(meta) == 6
    print("SELF-TEST PASSED (synthetic FC, CBIG reshape, NaN/Inf exclusion; no real MRI was processed).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-subject CoRR-HNU surface FC pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Folder containing the five downloaded files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write outputs")
    parser.add_argument("--self-test", action="store_true", help="Run a quick mathematical test using synthetic arrays")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    folder = args.data_dir.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"Data folder does not exist: {folder}")
    output = args.output_dir or (folder / "results_subj01")
    try:
        run(folder, output)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Please copy the complete error message into our chat so we can fix it.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
