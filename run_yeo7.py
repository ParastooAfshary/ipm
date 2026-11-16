#!/usr/bin/env python3
"""Yeo 2011 seven-network analysis of CoRR-HNU subj01 fsaverage5 surface fMRI.

Uses the official CBIG fsaverage5 surface annotation files, not the volumetric
Yeo atlas. This is group-reference network analysis, NOT MS-HBM personalized
parcellation. It reuses the exact fMRI and censor handling in run_fc.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from run_fc import locate_inputs, read_surface_nifti, read_censor

BASE_URL = (
    'https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/'
    'stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/'
    '1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/'
    'fsaverage5/label/'
)
ANNOT_FILENAME = '{hemi}.Yeo2011_7Networks_N1000.annot'
NETWORK_NAMES = [
    'Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention',
    'Limbic', 'Frontoparietal', 'DefaultMode',
]


def download_annotations(folder: Path) -> dict[str, Path]:
    import requests
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for hemi in ('lh', 'rh'):
        filename = ANNOT_FILENAME.format(hemi=hemi)
        path = folder / filename
        if not path.exists() or path.stat().st_size < 1000:
            url = BASE_URL + filename
            print(f'      Downloading official CBIG surface atlas: {filename}', flush=True)
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f'Could not download {url}. Check your connection or download this file '
                    f'manually into {folder}. Original error: {exc}'
                ) from exc
            if len(response.content) < 1000:
                raise RuntimeError(f'Atlas download looks too small ({len(response.content)} bytes): {url}')
            path.write_bytes(response.content)
        result[hemi] = path
    return result


def decode_label(x) -> str:
    return x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x)


def annotation_to_network_ids(path: Path) -> tuple[np.ndarray, dict]:
    from nibabel.freesurfer.io import read_annot
    labels, _ctab, raw_names = read_annot(str(path), orig_ids=False)
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (10242,):
        raise ValueError(f'{path.name}: expected 10242 fsaverage5 vertices, got {labels.shape}')
    names = [decode_label(n) for n in raw_names]
    used = [i for i, name in enumerate(names)
            if np.any(labels == i) and not any(w in name.lower() for w in ('unknown', 'medial_wall'))]
    if len(used) != 7:
        raise ValueError(
            f'{path.name}: expected exactly seven non-background networks; '
            f'found {len(used)}: {[(i, names[i]) for i in used]}. '
            'Please send this error before proceeding.'
        )
    # Prefer explicit network numbers, otherwise annotation color-table order.
    # For original Yeo 7-network .annot, the seven networks are ordered 1..7.
    numbered = {}
    for i in used:
        match = re.search(r'7Networks[_-]([1-7])(?:$|[_-])', names[i], flags=re.I)
        if match:
            numbered[i] = int(match.group(1))
    if len(numbered) == 7 and set(numbered.values()) == set(range(1, 8)):
        index_to_id = numbered
        mapping_mode = 'explicit_network_number'
    else:
        index_to_id = {i: rank for rank, i in enumerate(sorted(used), start=1)}
        mapping_mode = 'annotation_color_table_order'
    ids = np.zeros(10242, dtype=np.int8)
    for idx, network_id in index_to_id.items():
        ids[labels == idx] = network_id
    present = set(np.unique(ids).tolist())
    if not set(range(1, 8)).issubset(present) or not present.issubset(set(range(8))):
        raise ValueError(f'{path.name}: unexpected network ids {np.unique(ids)}')
    details = {'annotation_labels': {str(i): names[i] for i in used}, 'mapping_mode': mapping_mode,
               'vertices_by_network': {NETWORK_NAMES[i-1]: int((ids == i).sum()) for i in range(1, 8)},
               'unlabeled_vertices': int((ids == 0).sum())}
    return ids, details


def compute_networks(lh: np.ndarray, rh: np.ndarray, keep: np.ndarray,
                     lh_ids: np.ndarray, rh_ids: np.ndarray):
    if lh.shape != rh.shape or lh.shape[0] != 10242 or lh.shape[1] != len(keep):
        raise ValueError(f'Unexpected fMRI/censor shapes: {lh.shape}, {rh.shape}, {len(keep)}')
    all_data = np.concatenate([lh, rh], axis=0)
    ids = np.concatenate([lh_ids, rh_ids])
    finite = np.isfinite(all_data).all(axis=1)
    varying = np.zeros(len(finite), dtype=bool)
    varying[finite] = np.std(all_data[finite], axis=1, dtype=np.float64) > 1e-8
    usable = finite & varying & (ids > 0)
    X = all_data[usable][:, keep].astype(np.float64)
    group = ids[usable]
    # Each vertex's time course is centered and scaled to unit L2 norm.
    centered = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    good = norms > 1e-10
    X, centered, norms, group = X[good], centered[good], norms[good], group[good]
    Z = centered / norms[:, None]
    nframes = X.shape[1]
    means, sums, counts = [], [], []
    for i in range(1, 8):
        mask = group == i
        count = int(mask.sum())
        if count < 2:
            raise ValueError(f'Network {NETWORK_NAMES[i-1]} has only {count} valid vertices')
        counts.append(count)
        means.append(X[mask].mean(axis=0))
        sums.append(Z[mask].sum(axis=0))
    ts = np.column_stack(means)  # retained_frames x 7
    fc = np.corrcoef(ts, rowvar=False)
    if not np.isfinite(fc).all():
        raise ValueError('Yeo network-mean FC has NaN/Inf')
    # Mean pairwise vertex correlations (diagonal excludes self-correlations).
    # This algebraic calculation avoids storing a 20k x 20k matrix.
    avg_vertex_fc = np.empty((7, 7), dtype=np.float64)
    for i in range(7):
        for j in range(7):
            product = float(np.dot(sums[i], sums[j]))
            if i == j:
                avg_vertex_fc[i, j] = (product - counts[i]) / (counts[i] * (counts[i] - 1))
            else:
                avg_vertex_fc[i, j] = product / (counts[i] * counts[j])
    if not (np.isfinite(avg_vertex_fc).all() and np.isfinite(ts).all()):
        raise ValueError('Invalid values in Yeo network analysis')
    return ts, fc, avg_vertex_fc, counts, {
        'total_surface_vertices': int(len(usable)),
        'finite_varying_labeled_vertices': int(len(X)),
        'excluded_nonfinite': int((~finite).sum()),
        'excluded_constant': int((finite & ~varying).sum()),
        'excluded_unlabeled_after_qc': int((finite & varying & (ids == 0)).sum()),
        'frames_retained': int(nframes),
    }


def save_matrix(path: Path, mat: np.ndarray) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['network'] + NETWORK_NAMES)
        for name, row in zip(NETWORK_NAMES, mat):
            w.writerow([name] + [f'{float(x):.9g}' for x in row])


def save_heatmap(mat: np.ndarray, title: str, path: Path, within: bool = False) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest')
    ax.set_xticks(range(7), NETWORK_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(7), NETWORK_NAMES)
    ax.set_title(title)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(mat[i,j]) > 0.7 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if within:
        fig.text(0.5, 0.015, 'Diagonal = mean pairwise vertex correlation (self-pairs excluded)',
                 ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def self_test() -> None:
    rng = np.random.default_rng(5)
    nframes = 70
    lh = rng.normal(size=(10242, nframes)).astype('float32')
    rh = rng.normal(size=(10242, nframes)).astype('float32')
    labels = np.tile(np.arange(1, 8), 1464)[:10242].astype('int8')
    keep = np.ones(nframes, dtype=bool)
    keep[10:12] = False
    ts, fc, pair, counts, qc = compute_networks(lh, rh, keep, labels, labels)
    assert ts.shape == (68, 7) and fc.shape == (7, 7) and pair.shape == (7, 7)
    assert np.allclose(fc, fc.T) and np.allclose(np.diag(fc), 1)
    assert np.allclose(pair, pair.T)
    assert sum(counts) == 20484 and qc['frames_retained'] == 68
    # Compare formula to brute-force mean off-diagonal vertex correlation.
    tiny = rng.normal(size=(5, 50))
    tiny -= tiny.mean(axis=1, keepdims=True)
    tiny /= np.linalg.norm(tiny, axis=1, keepdims=True)
    brute = np.corrcoef(tiny).sum() - 5
    formula = (np.dot(tiny.sum(axis=0), tiny.sum(axis=0)) - 5)
    assert np.allclose(brute, formula, atol=1e-12)
    print('SELF-TEST PASSED (synthetic data only; official atlas not downloaded).')


def main() -> None:
    parser = argparse.ArgumentParser(description='Yeo7 official CBIG fsaverage5 network analysis')
    parser.add_argument('--data-dir', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = args.data_dir.resolve()
    output = (args.output_dir or (root / 'results_subj01' / 'yeo7')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        print('[1/5] Reading fsaverage5 fMRI and censor file ...', flush=True)
        files = locate_inputs(root)
        lh, _ = read_surface_nifti(files['lh_fs5'])
        rh, _ = read_surface_nifti(files['rh_fs5'])
        keep, note = read_censor(files['censor'], lh.shape[1])
        print(f'      Retained {int(keep.sum())}/{len(keep)} frames. {note}', flush=True)
        print('[2/5] Loading original Yeo2011 seven-network surface annotations ...', flush=True)
        annotations = download_annotations(root / 'yeo7_atlas')
        lh_ids, lh_details = annotation_to_network_ids(annotations['lh'])
        rh_ids, rh_details = annotation_to_network_ids(annotations['rh'])
        if lh_details['mapping_mode'] != rh_details['mapping_mode']:
            raise ValueError('LH/RH annotations use different network mapping modes')
        if lh_details['mapping_mode'] == 'annotation_color_table_order':
            left_names = list(lh_details['annotation_labels'].values())
            right_names = list(rh_details['annotation_labels'].values())
            if left_names != right_names:
                raise ValueError('LH/RH annotation names differ: cannot safely assume matching network order')
        print('[3/5] Extracting seven network mean signals and vertex-pair correlations ...', flush=True)
        ts, fc, pair, counts, qc = compute_networks(lh, rh, keep, lh_ids, rh_ids)
        print('      Network counts:', dict(zip(NETWORK_NAMES, counts)), flush=True)
        print('[4/5] Saving matrices and heatmaps ...', flush=True)
        save_matrix(output / 'yeo7_network_mean_fc.csv', fc)
        save_matrix(output / 'yeo7_mean_vertex_pairwise_fc.csv', pair)
        np.save(output / 'yeo7_network_timeseries.npy', ts)
        with (output / 'yeo7_network_timeseries.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['original_frame_index'] + NETWORK_NAMES)
            for frame, values in zip(np.flatnonzero(keep), ts):
                w.writerow([int(frame)] + [f'{float(v):.9g}' for v in values])
        save_heatmap(fc, 'Yeo7 | Correlation of network-mean time series', output / 'yeo7_network_mean_fc_heatmap.png')
        save_heatmap(pair, 'Yeo7 | Mean pairwise vertex correlation', output / 'yeo7_mean_vertex_pairwise_fc_heatmap.png', within=True)
        print('[5/5] Saving QC and provenance ...', flush=True)
        report = {
            'subject': 'CoRR-HNU subj01 session1',
            'atlas': 'Yeo 2011 group-reference seven networks, original CBIG fsaverage5 surface annotations',
            'is_individual_specific_parcellation': False,
            'frames_total': int(len(keep)), 'frames_retained': int(keep.sum()),
            'censor_convention': '0=high motion/outlier, 1=low motion/retained (CBIG documentation)',
            'networks': NETWORK_NAMES, 'valid_vertices_per_network': dict(zip(NETWORK_NAMES, counts)),
            'annotation_source_urls': {h: BASE_URL + ANNOT_FILENAME.format(hemi=h) for h in ('lh','rh')},
            'annotation_details': {'lh': lh_details, 'rh': rh_details},
            'qc': qc,
            'interpretation': (
                'Network-mean FC is based on mean time courses. Mean pairwise vertex FC '
                'is a separate statistic; its diagonal excludes self-correlations. '
                'Group Yeo networks are not subject-specific Kong MS-HBM networks. '
                'Spatial matching between CBIG fMRI and CBIG surface atlas is assumed and should be independently verified.'
            ),
        }
        (output / 'yeo7_qc.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'\nSUCCESS! Yeo7 outputs saved in: {output}', flush=True)
        print('Please send yeo7_qc.json and yeo7_network_mean_fc_heatmap.png for review.')
    except Exception as exc:
        print(f'\nERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('Please send the complete error so we can fix it.', file=sys.stderr)
        raise SystemExit(1) from exc

if __name__ == '__main__':
    main()
