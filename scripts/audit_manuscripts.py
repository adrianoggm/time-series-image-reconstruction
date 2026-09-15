"""Read-only manuscript audit; writes evidence only under --out.

Run with the scientific environment and the current checkout as working directory:
python -m scripts.audit_manuscripts --data /path/data/WISDM --results /path/results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENC = {'RP': ('recurrence_plot', 'rec'), 'GASF': ('GAF', 'gaf'),
       'GADF': ('GADF', 'gadf'), 'MTF': ('MTF', 'mtf')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'docs/audit')
    ap.add_argument('--compare-root', type=Path, default=None,
                    help='root of the OTHER checkout to diff sources and stats against; '
                         'ignored when it resolves to this one')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from src.fidelity_metrics import all_metrics
    from scripts._make_error_tables import per_window
    maintex = (ROOT / 'doc/main2.tex').read_text(encoding='utf8')
    supp = (ROOT / 'doc/supplementary.tex').read_text(encoding='utf8')
    evidence = {'inputs': {}, 'folds': {}, 'pooled': {}, 'checks': [], 'code_differences': []}
    def check(name, ok, **detail):
        evidence['checks'].append(dict(name=name, ok=bool(ok), **detail))
    def table(text, label):
        pos = text.index('\\label{' + label + '}')
        return text[pos:text.index('\\end{table', pos)]
    by_protocol = {}
    for protocol in ('loto', 'loso'):
        pieces, subjects, labels, recs = [], [], [], {k: [] for k in ENC}
        fold_results = {k: [] for k in ENC}
        for fold in range(3):
            tail = Path(f'sampling_{protocol}/3-fold/fold-{fold}/test')
            raw = a.data / 'tseries/recurrence_plot' / tail
            x = np.load(raw / 'validation_data.npy')
            s = np.load(raw / 'sj_validation_data.npy').reshape(-1)
            y = np.load(raw / 'y_validation_data.npy')
            y = y.argmax(1) if y.ndim == 2 else y.reshape(-1)
            train_s = np.load(raw.parent / 'train/sj_training_data.npy').reshape(-1)
            check(f'{protocol}/fold{fold}/finite_original', np.isfinite(x).all())
            if protocol == 'loso':
                check(f'{protocol}/fold{fold}/subject_disjoint', not set(s) & set(train_s))
            evidence['folds'][f'{protocol}/{fold}'] = dict(n=len(x), subjects=len(set(s)), classes=np.bincount(y).tolist())
            pieces.append(x); subjects.append(s); labels.append(y)
            for enc, (folder, suffix) in ENC.items():
                p = a.data / 'tseries' / folder / tail / f'X_all_{suffix}.npy'
                r = np.load(p)
                evidence['inputs'][str(p)] = dict(sha256=sha(p), shape=list(r.shape))
                if r.shape != x.shape:
                    r = r.transpose(0, 2, 1)
                check(f'{protocol}/fold{fold}/{enc}/finite_shape', r.shape == x.shape and np.isfinite(r).all())
                recs[enc].append(r)
                pw = per_window(x, r)
                fold_results[enc].append({k: float(v[0].mean()) for k, v in pw.items()})
        x, s, y = np.concatenate(pieces), np.concatenate(subjects), np.concatenate(labels)
        recs = {k: np.concatenate(v) for k, v in recs.items()}
        hashes = [hashlib.sha256(w.tobytes() + str(sj).encode() + str(lab).encode()).hexdigest() for w, sj, lab in zip(x, s, y)]
        check(f'{protocol}/unique_windows', len(set(hashes)) == 8172, actual=len(set(hashes)))
        evidence['folds'][protocol] = fold_results
        by_protocol[protocol] = (hashes, recs)
        pooled = {}
        for enc, r in recs.items():
            pw = per_window(x, r)
            m = all_metrics(x, r)
            m.update({k: v[0] for k, v in pw.items() if k != 'MRE'})
            pooled[enc] = {k: dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v))) for k, v in m.items()}
            pooled[enc]['axes'] = {k: [dict(mean=float(col.mean()), std=float(col.std())) for col in v[1].T] for k, v in pw.items()}
            pooled[enc]['subject_mae'] = float(np.mean([pw['MAE'][0][s == sj].mean() for sj in np.unique(s)]))
            if protocol == 'loto':
                for metric, d in [('MAE', 2 if enc == 'MTF' else 4), ('MSE', 2 if enc == 'MTF' else 4), ('Pearson', 5)]:
                    row = next(z for z in table(maintex, 'tab:pointwise_pooled').splitlines() if z.startswith(enc + ' '))
                    expected = f'{pooled[enc][metric]["mean"]:.{d}f}'
                    check(f'main/pointwise/{enc}/{metric}', expected in row, expected=expected, row=row)
                saved = json.loads((ROOT / 'results/_stats/fidelity_metrics.json').read_text())
                evidence['saved_fidelity_schema'] = list(saved)
        evidence['pooled'][protocol] = pooled
        evidence['range_exclusions_' + protocol] = int(np.sum(np.ptp(x, axis=1) <= 1e-8))
        print('Computed all reconstruction metrics:', protocol, flush=True)
    h0, r0 = by_protocol['loto']; h1, r1 = by_protocol['loso']
    check('same_window_sets', set(h0) == set(h1))
    order = {h: i for i, h in enumerate(h1)}
    idx = [order[h] for h in h0]
    for enc in ENC:
        diff = np.abs(r0[enc] - r1[enc][idx])
        check(f'protocol_independent/{enc}', float(diff.max()) < 1e-8, max_abs=float(diff.max()), differing_windows=int(np.sum(diff.max(axis=(1, 2)) > 1e-8)))

    # Compare every per-class table body against the live result JSONs.
    from scripts import emit_classifier_tables as emitter
    emitter._load = lambda clf, stype, sampling, fold: json.loads((a.results / stype / clf / sampling / f'fold-{fold}.json').read_text())
    names = {'gru': 'gru', 'knn': 'knn', 'cnn': 'cnn', 'randomforest': 'rf', 'svc': 'svm', 'lstm': 'lstm'}
    for clf, label in names.items():
        body = table(supp, f'tab:{label}_classifiers')
        actual = emitter.emit(clf, 'loto')
        for line in actual.splitlines():
            if '&' in line:
                check(f'supp/perclass/{clf}/{line.split("&")[0].strip()}', re.sub(r'\s+', '', line) in re.sub(r'\s+', '', body), expected=line)

    accs = {}
    for protocol in ('loto', 'loso'):
        accs[protocol] = {}
        for clf in names:
            d = {}
            original_reports = []
            for enc in ('RP', 'GAF', 'GADF', 'MTF'):
                js = [json.loads((a.results / enc / clf / protocol / f'fold-{f}.json').read_text()) for f in range(3)]
                original_reports.append([j['evaluations']['original'] for j in js])
                vals = [j['evaluations']['reconstructed']['accuracy'] for j in js]
                d[enc] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)))
            check(f'original_cache_consistent/{protocol}/{clf}', all(v == original_reports[0] for v in original_reports))
            vals = [v['accuracy'] for v in original_reports[0]]
            d['Original'] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)))
            accs[protocol][clf] = d
            if protocol == 'loso':
                row = next(z for z in table(supp, 'tab:gadf_summary_perfold_subject_out').splitlines() if z.startswith({'svc':'SVM', 'randomforest':'RF'}.get(clf, clf.upper()) + ' '))
                for enc in ('Original', 'RP', 'GAF', 'GADF', 'MTF'):
                    expected = f'{d[enc]["mean"]:.3f}\\pm{d[enc]["std"]:.3f}'
                    check(f'supp/loso_acc/{clf}/{enc}', expected in row, expected=expected, row=row)
    evidence['accuracy'] = accs
    # Compare this checkout against the other one, read-only. Skipped unless
    # --compare-root names a DIFFERENT tree: comparing a checkout with itself
    # silently reports "no differences" and hides a real desynchronisation.
    cmp_root = a.compare_root.resolve() if a.compare_root else None
    if cmp_root is None or cmp_root == ROOT:
        evidence['cross_checkout_comparison'] = 'skipped (no distinct --compare-root)'
    else:
        evidence['cross_checkout_comparison'] = str(cmp_root)
        for folder in ('src', 'scripts'):
            for p in (ROOT / folder).glob('*.py'):
                other = cmp_root / folder / p.name
                if other.exists() and p.read_text(encoding='utf8') != other.read_text(encoding='utf8'):
                    evidence['code_differences'].append(str(p.relative_to(ROOT)))
        for p in (ROOT / 'results/_stats').glob('*.json'):
            other = cmp_root / 'results/_stats' / p.name
            if other.exists():
                check(f'stats_sync/{p.name}',
                      json.loads(p.read_text()) == json.loads(other.read_text()))
    (a.out / 'numerical_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf8')
    bad = [c for c in evidence['checks'] if not c['ok']]
    print(json.dumps({'checks': len(evidence['checks']), 'failed': bad, 'code_differences': evidence['code_differences']}, indent=2))


if __name__ == '__main__':
    main()
