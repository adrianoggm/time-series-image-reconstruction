"""M7 classification utility (roadmap #89): do reconstructions keep what a classifier needs?

For each labelled dataset a seeded sample of windows is reconstructed once per
technique from its 8-bit image. Classifiers are trained on ORIGINAL windows only
(grouped 3-fold: no subject/patient/event/recording on both sides) and tested on
the original and on every reconstruction of the same held-out windows, as in
the WISDM paper. Two classifiers: a random forest on the flattened window and a
small 1-D CNN (inputs standardised per channel with training statistics).

Metrics: accuracy and macro-F1 (multiclass); macro-F1 and macro ROC-AUC for the
multilabel PTB-XL superclasses. For multiclass data the per-group accuracy
difference (original minus reconstruction) gets a 95% cluster-bootstrap
interval; equivalence at the paper's margin means the whole interval lies in
[-0.01, 0.01].

    python -m scripts._classification --length 128 --max-windows 6000
Writes results/_classification/summary.json and summary.md.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATASETS = {"wisdm": {}, "pamap2": {}, "esc50": {"target_fs": 8000}, "ptbxl": {"fs": 100, "max_records": 3000},
            "stead": {"n_per_class": 1000}, "emg_gestures": {}}
TECHS = ("RP_HIRATA", "RP_HIRATA_SIGNED", "RP_UNSIGNED", "RP", "GASF", "GADF", "MTF", "XIRP")
OUT = Path("results/_classification")
B = 2000


def reconstruct(window: np.ndarray) -> np.ndarray:
    """(T, C) -> (techniques, T, C) reconstructions from 8-bit images; flat channels are copied."""
    from threadpoolctl import threadpool_limits

    from src.representations import REPRESENTATIONS

    out = np.repeat(window[None].astype(np.float32), len(TECHS), axis=0)
    with threadpool_limits(1):
        for c in range(window.shape[1]):
            x = window[:, c].astype(float)
            if x.max() - x.min() < 1e-12:
                continue
            mm = (float(x.max()), float(x.min()))
            for k, name in enumerate(TECHS):
                rep = REPRESENTATIONS[name]
                try:
                    out[k, :, c] = rep.decode(rep.from_uint8(rep.to_uint8(rep.encode(x, mm))), mm)
                except Exception:
                    out[k, :, c] = np.nan
    return out


def cnn(n_in, n_ch, n_out, multilabel, seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((n_in, n_ch)),
        tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_out, activation="sigmoid" if multilabel else "softmax")])
    m.compile("adam", "binary_crossentropy" if multilabel else "sparse_categorical_crossentropy")
    return m


def scores(y, proba, multilabel):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if multilabel:
        pred = (proba >= 0.5).astype(int)
        keep = (y.sum(0) > 0) & (y.sum(0) < len(y))
        return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "macro_auc": float(roc_auc_score(y[:, keep], proba[:, keep], average="macro"))}
    pred = proba.argmax(1)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro"))}


def group_diff_ci(correct_orig, correct_rec, groups, rng):
    uniq, inv = np.unique(groups, return_inverse=True)
    d = np.array([correct_orig[inv == g].mean() - correct_rec[inv == g].mean() for g in range(len(uniq))])
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--max-windows", type=int, default=6000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier

    from src.datasets import get, group_kfold, make_windows

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary["_design"] = {"techniques": TECHS, "classifiers": ["rf", "cnn"], "length": args.length,
                          "max_windows": args.max_windows, "folds": args.folds, "epochs": args.epochs, "seed": 0,
                          "datasets": DATASETS, "doc": __doc__.split("\n\n")[1].strip()}
    rng = np.random.default_rng(0)
    for ds in args.datasets.split(","):
        t0 = time.perf_counter()
        adapter = get(ds, **DATASETS[ds])
        w = make_windows(adapter, args.length, dtype=np.float32)
        idx = np.sort(np.random.default_rng(0).choice(len(w.X), size=min(args.max_windows, len(w.X)), replace=False))
        X, y, groups = w.X[idx], w.y[idx], w.groups[idx]
        multilabel = y.ndim == 2
        sub = type(w)(X, y, groups, w.rec_id[idx], w.start[idx], w.meta)
        folds = group_kfold(sub, n_folds=args.folds)
        with ProcessPoolExecutor(args.workers) as ex:
            recs = np.stack(list(ex.map(reconstruct, list(X), chunksize=8)), axis=1)   # (techs, N, T, C)
        n_fail = {t: int(np.isnan(recs[k]).any(axis=(1, 2)).sum()) for k, t in enumerate(TECHS)}
        recs = np.nan_to_num(recs)
        classes = w.meta["classes"]
        n_out = len(classes)
        res = {"windows": int(len(X)), "channels": int(X.shape[2]), "classes": n_out, "multilabel": multilabel,
               "groups": int(len(np.unique(groups))), "reconstruction_failures": n_fail}
        for clf_name in ("rf", "cnn"):
            per = {"original": []} | {t: [] for t in TECHS}
            correct = {"original": np.zeros(len(X), bool)} | {t: np.zeros(len(X), bool) for t in TECHS}
            for fi, (tr, te) in enumerate(folds):
                mu = X[tr].mean(axis=(0, 1), keepdims=True)
                sd = X[tr].std(axis=(0, 1), keepdims=True) + 1e-12
                if clf_name == "rf":
                    model = RandomForestClassifier(n_estimators=300, n_jobs=args.workers, random_state=0)
                    model.fit(((X[tr] - mu) / sd).reshape(len(tr), -1), y[tr])

                    def predict(Z, model=model, mu=mu, sd=sd, multilabel=multilabel):
                        p = model.predict_proba(((Z - mu) / sd).reshape(len(Z), -1))
                        if not multilabel:
                            return p
                        # one output per label; a label never positive in training has a single column
                        return np.stack([q[:, list(c).index(1)] if 1 in c else np.zeros(len(q))
                                         for q, c in zip(p, model.classes_, strict=True)], 1)
                else:
                    import tensorflow as tf

                    model = cnn(X.shape[1], X.shape[2], n_out, multilabel, seed=fi)
                    stop = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
                    model.fit((X[tr] - mu) / sd, y[tr], epochs=args.epochs, batch_size=64, validation_split=0.1,
                              callbacks=[stop], verbose=0)

                    def predict(Z, model=model, mu=mu, sd=sd):
                        return model.predict((Z - mu) / sd, verbose=0)
                for name, Z in [("original", X[te])] + [(t, recs[k][te]) for k, t in enumerate(TECHS)]:
                    proba = predict(Z)
                    per[name].append(scores(y[te], proba, multilabel))
                    if not multilabel:
                        correct[name][te] = proba.argmax(1) == y[te]
            entry = {}
            for name, lst in per.items():
                entry[name] = {m: float(np.mean([s[m] for s in lst])) for m in lst[0]}
                if not multilabel and name != "original":
                    diff, ci = group_diff_ci(correct["original"], correct[name], groups, rng)
                    entry[name] |= {"group_acc_diff": diff, "ci95": ci, "equivalent_0.01": bool(-0.01 <= ci[0] and ci[1] <= 0.01)}
            res[clf_name] = entry
        res["wall_s"] = time.perf_counter() - t0
        summary[ds] = res
        path.write_text(json.dumps(summary, indent=1) + "\n")
        key = "macro_f1" if multilabel else "accuracy"
        print(f"{ds:13} {len(X)} win x {X.shape[2]} ch, {res['groups']} groups, {res['wall_s']:.0f} s | {key} RF/CNN: "
              + "  ".join(f"{n} {res['rf'][n][key]:.3f}/{res['cnn'][n][key]:.3f}" for n in ("original",) + TECHS), flush=True)
    write_md(summary)


def write_md(summary):
    from scripts._benchmark import LABELS

    rows = ["# M7 classification: classifiers trained on original windows, tested on reconstructions", "",
            "Accuracy for multiclass data, macro-F1 for multilabel PTB-XL. Grouped 3-fold. "
            "Mark ✓: per-group accuracy difference within ±0.01 (95% cluster bootstrap).", ""]
    for clf in ("rf", "cnn"):
        rows += [f"## {clf.upper()}", "", "| Dataset | Original | " + " | ".join(LABELS[t] for t in TECHS) + " |",
                 "|---|---|" + "---|" * len(TECHS)]
        for ds, r in summary.items():
            if ds.startswith("_"):
                continue
            key = "macro_f1" if r["multilabel"] else "accuracy"
            cells = [f"{r[clf][t][key]:.3f}" + (" ✓" if r[clf][t].get("equivalent_0.01") else "") for t in TECHS]
            rows.append(f"| {ds} | {r[clf]['original'][key]:.3f} | " + " | ".join(cells) + " |")
        rows.append("")
    (OUT / "summary.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    main()
