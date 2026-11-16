#!/usr/bin/env python3
r"""IPM preflight scientific/technical audit for CoRR-HNU subj01.

Run from ipm-1sub with the existing virtual environment:
    .venv\Scripts\python.exe ipm_preflight_audit.py

Reads original fsaverage5 MRI, censor, Yeo annotation and previous outputs.
Does NOT change source MRI, existing results or scripts.
Writes: results_subj01/preflight_audit.json and preflight_audit.txt

Notes: tests are technical checks and sensitivity diagnostics, not independent
clinical/scientific validation, test-retest reliability or proof of anatomy.
"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

NETWORK_NAMES = ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Frontoparietal", "DefaultMode"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def matrix_from_csv(path: Path) -> np.ndarray:
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    return np.asarray([[float(x) for x in row[1:]] for row in rows[1:]], dtype=np.float64)


def corr7(ts: np.ndarray) -> np.ndarray:
    return np.corrcoef(ts, rowvar=False)


def within_network_correlation(data: np.ndarray, ids: np.ndarray, keep: np.ndarray, network_id: int) -> dict:
    good = (ids == network_id) & np.isfinite(data).all(axis=1)
    x = data[good][:, keep].astype(np.float64)
    if len(x) < 2:
        return {"n_vertices": int(len(x)), "mean_pairwise_r": None}
    x -= x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1)
    x = x[norms > 1e-10]
    norms = norms[norms > 1e-10]
    if len(x) < 2:
        return {"n_vertices": int(len(x)), "mean_pairwise_r": None}
    sum_z = (x / norms[:, None]).sum(axis=0)
    n = len(x)
    return {"n_vertices": n, "mean_pairwise_r": float((np.dot(sum_z, sum_z) - n) / (n * (n - 1)))}


def main() -> int:
    import nibabel as nib
    from nilearn.datasets import fetch_atlas_surf_destrieux
    from run_fc import locate_inputs, read_surface_nifti, read_censor
    from run_yeo7 import annotation_to_network_ids, ANNOT_FILENAME

    root = Path(__file__).resolve().parent
    out = root / 'results_subj01'
    yeo = out / 'yeo7'
    report = {"purpose": "preflight audit of current single-subject atlas-based FC", "checks": {}, "diagnostics": {}, "caveats": [
        "Group Yeo7 and anatomical Destrieux are NOT subject-specific Kong MS-HBM parcellations.",
        "One run and a split-half diagnostic do not establish independent test-retest reliability.",
        "Global signal regression and complete preprocessing cannot be established from filenames alone.",
        "Matching medial-wall masks supports surface indexing but does not independently prove every anatomical vertex is correctly registered.",
        "fsaverage5 sm6 data are not sufficient for high-resolution Ladwig patch replication.",
    ]}
    checks = report['checks']
    def check(name: str, cond: bool, detail: str = '') -> None:
        checks[name] = {"passed": bool(cond), "detail": detail}
        print(f"  {'PASS' if cond else 'FAIL'} {name}" + (f" | {detail}" if detail else ''), flush=True)

    for path in (out / 'roi_timeseries.npy', out / 'roi_metadata.csv',
                 yeo / 'yeo7_network_timeseries.npy',
                 yeo / 'yeo7_network_mean_fc.csv',
                 yeo / 'yeo7_mean_vertex_pairwise_fc.csv'):
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; please run previous pipelines first')

    inputs = locate_inputs(root)
    print('[1/6] Re-reading original MRI and censor...', flush=True)
    lh, lh_shape = read_surface_nifti(inputs['lh_fs5'])
    rh, rh_shape = read_surface_nifti(inputs['rh_fs5'])
    keep, censor_note = read_censor(inputs['censor'], lh.shape[1])
    check('Same time dimension LH/RH/censor', lh.shape == rh.shape == (10242, len(keep)))
    check('Censor has 286 retained of 296 frames', len(keep) == 296 and int(keep.sum()) == 286,
          f'{int(keep.sum())}/{len(keep)}; {censor_note}')
    report['diagnostics']['censored_frame_numbers_1_based'] = (np.flatnonzero(~keep) + 1).tolist()
    report['diagnostics']['nifti_shapes'] = {'lh': list(lh_shape), 'rh': list(rh_shape)}

    print('[2/6] Checking vertex order against Yeo annotation medial wall...', flush=True)
    ids = {}
    mask_overlap = {}
    for hemi, data in (('lh', lh), ('rh', rh)):
        annot = root / 'yeo7_atlas' / ANNOT_FILENAME.format(hemi=hemi)
        if not annot.is_file():
            raise FileNotFoundError(f'Missing {annot}; original run_yeo7.py should have downloaded it')
        label_ids, _details = annotation_to_network_ids(annot)
        ids[hemi] = label_ids
        bad = ~np.isfinite(data).all(axis=1)
        unlabeled = label_ids == 0
        overlap = int(np.count_nonzero(bad & unlabeled))
        mismatch = int(np.count_nonzero(bad ^ unlabeled))
        mask_overlap[hemi] = {'nonfinite_vertices': int(bad.sum()), 'unlabeled_yeo_vertices': int(unlabeled.sum()),
                              'intersection': overlap, 'xor_mismatch': mismatch}
        check(f'{hemi} nonfinite mask matches Yeo medial wall exactly', mismatch == 0,
              f'{overlap} overlap; {mismatch} mismatches')
    report['diagnostics']['medial_wall_mask_overlap'] = mask_overlap

    print('[3/6] Checking Destrieux ROI membership and time series against original MRI...', flush=True)
    atlas = fetch_atlas_surf_destrieux()
    maps = {'lh': np.asarray(atlas['map_left']).reshape(-1), 'rh': np.asarray(atlas['map_right']).reshape(-1)}
    with (out / 'roi_metadata.csv').open(encoding='utf-8-sig', newline='') as f:
        metadata = list(csv.DictReader(f))
    roi_ts = np.load(out / 'roi_timeseries.npy', allow_pickle=False)
    check('Saved ROI matrix shape 286 x 148', roi_ts.shape == (int(keep.sum()), len(metadata)) and len(metadata) == 148,
          str(roi_ts.shape))
    errors = []
    count_errors = []
    for col, row in enumerate(metadata):
        hemi = row['hemisphere']
        data = {'lh': lh, 'rh': rh}[hemi]
        good = np.isfinite(data).all(axis=1)
        selected = (maps[hemi] == int(row['atlas_index'])) & good
        n = int(selected.sum())
        if n != int(row['usable_vertices']):
            count_errors.append({'roi': row['roi'], 'expected': int(row['usable_vertices']), 'recomputed': n})
        if n:
            actual = data[selected][:, keep].mean(axis=0, dtype=np.float64)
            errors.append(float(np.max(np.abs(actual - roi_ts[:, col]))))
    max_err = max(errors) if errors else float('inf')
    check('ROI vertex counts reproduced from atlas and MRI', not count_errors,
          f'{len(count_errors)} discrepancies')
    check('ROI time series reproduced from original MRI', max_err < 1e-5,
          f'max absolute difference={max_err:.3g}')
    report['diagnostics']['roi_reconstruction'] = {'max_abs_difference': max_err, 'vertex_count_errors': count_errors[:10]}

    print('[4/6] Checking saved Yeo7 network means and matrices...', flush=True)
    yeo_ts = np.load(yeo / 'yeo7_network_timeseries.npy', allow_pickle=False)
    all_data = np.concatenate([lh, rh], axis=0)
    all_ids = np.concatenate([ids['lh'], ids['rh']])
    finite = np.isfinite(all_data).all(axis=1)
    varying = np.zeros(len(finite), dtype=bool)
    varying[finite] = np.std(all_data[finite], axis=1, dtype=np.float64) > 1e-8
    usable = finite & varying
    recomputed_means = np.column_stack([
        all_data[usable & (all_ids == i)][:, keep].mean(axis=0, dtype=np.float64)
        for i in range(1, 8)
    ])
    yeo_ts_error = float(np.max(np.abs(recomputed_means - yeo_ts)))
    check('Yeo7 network time series reproduced from original MRI',
          yeo_ts.shape == recomputed_means.shape and yeo_ts_error < 1e-5,
          f'max absolute difference={yeo_ts_error:.3g}')
    saved_fc = matrix_from_csv(yeo / 'yeo7_network_mean_fc.csv')
    fc_error = float(np.max(np.abs(saved_fc - corr7(yeo_ts))))
    check('Yeo7 mean-network FC reproduced from saved series', fc_error < 1e-7,
          f'max absolute difference={fc_error:.3g}')
    saved_pair = matrix_from_csv(yeo / 'yeo7_mean_vertex_pairwise_fc.csv')
    check('Yeo7 pairwise vertex FC symmetric and within [-1,1]',
          saved_pair.shape == (7,7) and np.allclose(saved_pair, saved_pair.T, atol=1e-7)
          and np.max(np.abs(saved_pair)) <= 1.000001)
    report['diagnostics']['yeo_reconstruction'] = {'network_ts_max_abs_difference': yeo_ts_error,
                                                    'network_fc_max_abs_difference': fc_error}

    print('[5/6] Sensitivity and split-half diagnostics (NOT test-retest validation)...', flush=True)
    n = yeo_ts.shape[0]
    half_a = corr7(yeo_ts[:n//2])
    half_b = corr7(yeo_ts[n//2:])
    iu = np.triu_indices(7, k=1)
    edge_pattern_r = float(np.corrcoef(half_a[iu], half_b[iu])[0,1])
    max_half_delta = float(np.max(np.abs(half_a[iu] - half_b[iu])))
    # A no-censor comparison is a sensitivity diagnostic, not a recommended final FC.
    uncensored_means = np.column_stack([
        all_data[usable & (all_ids == i)].mean(axis=0, dtype=np.float64)
        for i in range(1, 8)
    ])
    uncensored_fc = corr7(uncensored_means)
    censor_max_delta = float(np.max(np.abs(uncensored_fc[iu] - saved_fc[iu])))
    diag = {'split_half': {'n_first': int(n//2), 'n_second': int(n-n//2),
                           'edge_pattern_correlation': edge_pattern_r,
                           'max_abs_edge_difference': max_half_delta,
                           'DMN_DAN_first': float(half_a[6,2]), 'DMN_DAN_second': float(half_b[6,2]),
                           'DMN_Visual_first': float(half_a[6,0]), 'DMN_Visual_second': float(half_b[6,0])},
            'censor_sensitivity': {'max_abs_edge_difference_all_vs_kept': censor_max_delta,
                                   'DMN_DAN_all_frames': float(uncensored_fc[6,2]),
                                   'DMN_DAN_kept_frames': float(saved_fc[6,2])},
            'within_network_hemisphere_breakdown': {}}
    for i, name in enumerate(NETWORK_NAMES, 1):
        diag['within_network_hemisphere_breakdown'][name] = {
            h: within_network_correlation({'lh': lh, 'rh': rh}[h], ids[h], keep, i)
            for h in ('lh', 'rh')
        }
    report['diagnostics']['sensitivity'] = diag
    print(f'  INFO split-half 21-edge pattern r = {edge_pattern_r:.3f} (not independent reliability)')
    print(f'  INFO split-half max |delta r| = {max_half_delta:.3f}')
    print(f'  INFO censor sensitivity max |delta r| = {censor_max_delta:.3f}')
    print(f'  INFO DMN-DAN first/second = {half_a[6,2]:.3f} / {half_b[6,2]:.3f}')
    print(f'  INFO DMN-DAN with/without censor = {saved_fc[6,2]:.3f} / {uncensored_fc[6,2]:.3f}')
    print(f'  INFO Somatomotor within LH/RH = '
          f"{diag['within_network_hemisphere_breakdown']['Somatomotor']['lh']['mean_pairwise_r']:.3f} / "
          f"{diag['within_network_hemisphere_breakdown']['Somatomotor']['rh']['mean_pairwise_r']:.3f}")

    print('[6/6] Recording reproducibility fingerprints and writing audit...', flush=True)
    paths = dict(inputs)
    for h in ('lh','rh'):
        paths[f'yeo_annot_{h}'] = root / 'yeo7_atlas' / ANNOT_FILENAME.format(hemi=h)
    for fname in ('run_fc.py','run_yeo7.py'):
        if (root / fname).is_file():
            paths[fname] = root / fname
    report['sha256'] = {key: sha256(path) for key,path in paths.items()}
    report['all_technical_checks_passed'] = all(item['passed'] for item in checks.values())
    report['recommended_next_actions'] = [
        'Verify full preprocessing specification including global signal regression and TR from CBIG source.',
        'Review split-half and censor-sensitivity diagnostics; avoid significance claims from one run.',
        'Use official CBIG single-subject MS-HBM wrapper with original fsaverage6 files and compatible priors.',
        'For Ladwig-style fine LPFC patches obtain suitable high-resolution multi-session data rather than fsaverage5 sm6 alone.',
        'Use a matched structural T1 from the same participant before any individualized SimNIBS E-field modeling.',
    ]
    out.mkdir(parents=True, exist_ok=True)
    dest = out / 'preflight_audit.json'
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    text_lines = [
        'IPM preflight audit | CoRR-HNU subj01',
        f"Technical checks: {sum(x['passed'] for x in checks.values())}/{len(checks)} passed",
        *[f"{'PASS' if value['passed'] else 'FAIL'}: {name} | {value['detail']}" for name, value in checks.items()],
        '',
        f'Split-half network edge-pattern correlation (exploratory): {edge_pattern_r:.4f}',
        f'Split-half max edge delta: {max_half_delta:.4f}',
        f'Censor sensitivity max edge delta: {censor_max_delta:.4f}',
        f'DMN-DAN first/second: {half_a[6,2]:.4f} / {half_b[6,2]:.4f}',
        f'DMN-DAN with/without censor: {saved_fc[6,2]:.4f} / {uncensored_fc[6,2]:.4f}',
        'WARNING: Split-half is not test-retest; scientific validity is not established by this audit.',
    ]
    (out / 'preflight_audit.txt').write_text('\n'.join(text_lines) + '\n', encoding='utf-8')
    print('\n' + ('ALL TECHNICAL CHECKS PASSED' if report['all_technical_checks_passed'] else 'SOME TECHNICAL CHECKS FAILED'))
    print('Saved:', dest)
    print('Saved:', out / 'preflight_audit.txt')
    return 0 if report['all_technical_checks_passed'] else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
