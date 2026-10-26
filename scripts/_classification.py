"""M7 classification utility (roadmap #89): do reconstructions keep what a classifier needs?

For each labelled dataset a seeded sample of windows is reconstructed once per
technique from its 8-bit image. Classifiers are trained on ORIGINAL windows only
(grouped 3-fold: no subject/patient/event/recording on both sides) and tested on
the original and on every reconstruction of the same held-out windows, as in
the WISDM paper, with the paper's six classifiers: KNN (k = 5), SVM (RBF, C = 1)
and a random forest on the flattened window; LSTM, GRU (64 -> 32 units, dropout
0.2, as in the paper) and a small 1-D CNN on the (T, C) window. Inputs are
standardised per channel with training statistics. Window lengths 64, 128, 256.

Metrics: accuracy and macro-F1 (multiclass); macro-F1 and macro ROC-AUC for the
multilabel PTB-XL superclasses. For multiclass data the per-group accuracy
difference (original minus reconstruction) gets a 95% cluster-bootstrap
interval; equivalence at the paper's margin means the whole interval lies in
[-0.01, 0.01].

    python -m scripts._classification --lengths 64,128,256 --max-windows 6000
Writes results/_classification/summary.json and summary.md.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

DATASETS = {"wisdm": {}, "pamap2": {}, "esc50": {"target_fs": 8000}, "ptbxl": {"fs": 100, "max_records": 3000},
            "stead": {"n_per_class": 1000}, "emg_gestures": {}}
MAIN = tuple(DATASETS)
# ESC-10 (official 10-class subset) at audio-meaningful lengths: run with --out results/_classification_esc10
# --lengths 512,1024 --datasets esc10_2k,esc10_4k,esc10_8k. Silent (zero-padded) windows are dropped and accuracy
# is also scored per clip, averaging the class probabilities of its windows.
DATASETS |= {f"esc10_{k}k": {"adapter": "esc50", "target_fs": k * 1000, "esc10": True, "min_std": 1e-3, "per_clip": True}
             for k in (2, 4, 8)}
TECHS = ("RP_HIRATA", "RP_HIRATA_SIGNED", "RP_UNSIGNED", "RP", "GASF", "GADF", "MTF", "XIRP",
         "RP_SIPERS", "RP_SIPERS_SIGNED")      # the last two were added with --only (see main)
CLASSIFIERS = ("knn", "svm", "rf", "lstm", "gru", "cnn")
OUT = Path("results/_classification")
B = 2000
MIN_WINDOWS = 200      # blocks below this are kept in the JSON but left out of the tables (EMG at L = 256: 20 windows)


def reconstruct(window: np.ndarray, techs: tuple = TECHS) -> np.ndarray:
    """(T, C) -> (techniques, T, C) reconstructions from 8-bit images; flat channels are copied."""
    from threadpoolctl import threadpool_limits

    from src.representations import REPRESENTATIONS

    out = np.repeat(window[None].astype(np.float32), len(techs), axis=0)
    with threadpool_limits(1):
        for c in range(window.shape[1]):
            x = window[:, c].astype(float)
            if x.max() - x.min() < 1e-12:
                continue
            mm = (float(x.max()), float(x.min()))
            for k, name in enumerate(techs):
                rep = REPRESENTATIONS[name]
                try:
                    out[k, :, c] = rep.decode(rep.from_uint8(rep.to_uint8(rep.encode(x, mm))), mm)
                except Exception:
                    out[k, :, c] = np.nan
    return out


def sklearn_model(kind, multilabel, workers):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=workers)
    if kind == "svm":
        svc = SVC(C=1.0, kernel="rbf", probability=True, random_state=0)
        return OneVsRestClassifier(svc, n_jobs=workers) if multilabel else svc
    return RandomForestClassifier(n_estimators=300, n_jobs=workers, random_state=0)


def keras_model(kind, n_in, n_ch, n_out, multilabel, seed):
    import tensorflow as tf

    L = tf.keras.layers
    tf.keras.utils.set_random_seed(seed)
    body = {"cnn": [L.Conv1D(32, 5, padding="same", activation="relu"), L.MaxPooling1D(2),
                    L.Conv1D(64, 5, padding="same", activation="relu"), L.GlobalAveragePooling1D(),
                    L.Dense(64, activation="relu")],
            "lstm": [L.LSTM(64, return_sequences=True), L.Dropout(0.2), L.LSTM(32), L.Dropout(0.2)],
            "gru": [L.GRU(64, return_sequences=True), L.Dropout(0.2), L.GRU(32), L.Dropout(0.2)]}[kind]
    m = tf.keras.Sequential([L.Input((n_in, n_ch)), *body,
                             L.Dense(n_out, activation="sigmoid" if multilabel else "softmax")])
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
    ap.add_argument("--datasets", default=",".join(MAIN))
    ap.add_argument("--lengths", default="64,128,256")
    ap.add_argument("--classifiers", default=",".join(CLASSIFIERS))
    ap.add_argument("--max-windows", type=int, default=6000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)   # the stored runs used 20 (summary.json)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", default="", help="add these techniques to the existing results: classifiers are "
                    "retrained on the originals and each added technique is scored against the original accuracy "
                    "of that same refit (stored as ref_original); existing entries are left untouched")
    args = ap.parse_args()
    out = Path(args.out)
    only = tuple(args.only.split(",")) if args.only else ()
    techs = only or TECHS

    from src.datasets import get, group_kfold, make_windows

    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    clfs = args.classifiers.split(",")
    summary["_design"] = {"techniques": TECHS, "classifiers": CLASSIFIERS, "lengths": [64, 128, 256],
                          "max_windows": args.max_windows, "folds": args.folds, "epochs": args.epochs, "seed": 0,
                          "datasets": DATASETS, "doc": __doc__.split("\n\n")[1].strip()}
    rng = np.random.default_rng(0)
    for ds, length in [(d, int(n)) for n in args.lengths.split(",") for d in args.datasets.split(",")]:
        t0 = time.perf_counter()
        cfg = dict(DATASETS[ds])
        min_std, per_clip = cfg.pop("min_std", 0.0), cfg.pop("per_clip", False)
        adapter = get(cfg.pop("adapter", ds), **cfg)
        w = make_windows(adapter, length, dtype=np.float32)
        audible = np.flatnonzero(w.X.std(axis=(1, 2)) > min_std)
        idx = np.sort(np.random.default_rng(0).choice(audible, size=min(args.max_windows, len(audible)), replace=False))
        X, y, groups = w.X[idx], w.y[idx], w.groups[idx]
        multilabel = y.ndim == 2
        clips = w.rec_id[idx]
        sub = type(w)(X, y, groups, clips, w.start[idx], w.meta)
        folds = group_kfold(sub, n_folds=args.folds)
        # untracked; lets a crashed run resume without re-decoding
        cache = out / (f"recs_{ds}_L{length}" + ("_" + "_".join(only) if only else "") + ".npy")
        if cache.exists():
            recs = np.load(cache)
        else:
            with ProcessPoolExecutor(args.workers) as ex:
                recs = np.stack(list(ex.map(partial(reconstruct, techs=techs), list(X), chunksize=8)), axis=1)
            np.save(cache, recs)                                                        # (techs, N, T, C)
        n_fail = {t: int(np.isnan(recs[k]).any(axis=(1, 2)).sum()) for k, t in enumerate(techs)}
        recs = np.nan_to_num(recs)
        classes = w.meta["classes"]
        n_out = len(classes)
        done = summary.get(f"{ds}_L{length}", {})
        if only:
            res = done | {"reconstruction_failures": done.get("reconstruction_failures", {}) | n_fail}
        else:
            res = {"dataset": ds, "length": length, "windows": int(len(X)), "channels": int(X.shape[2]),
                   "classes": n_out, "multilabel": multilabel, "groups": int(len(np.unique(groups))),
                   "reconstruction_failures": n_fail}
        for clf_name in clfs:
            if clf_name in done and all(t in done[clf_name] for t in techs):
                res[clf_name] = done[clf_name]
                continue
            per = {"original": []} | {t: [] for t in techs}
            correct = {"original": np.zeros(len(X), bool)} | {t: np.zeros(len(X), bool) for t in techs}
            for fi, (tr, te) in enumerate(folds):
                mu = X[tr].mean(axis=(0, 1), keepdims=True)
                sd = X[tr].std(axis=(0, 1), keepdims=True) + 1e-12
                # trained models are kept, so a technique added later only needs predictions (untracked, large).
                # joblib/h5 are only ever loaded from files this script wrote locally, never from elsewhere.
                saved = out / "models" / f"{ds}_L{length}" / f"{clf_name}_fold{fi}"
                saved.parent.mkdir(parents=True, exist_ok=True)
                if clf_name in ("knn", "svm", "rf"):
                    import joblib

                    if saved.with_suffix(".joblib").exists():
                        model = joblib.load(saved.with_suffix(".joblib"))
                    else:
                        model = sklearn_model(clf_name, multilabel, args.workers)
                        model.fit(((X[tr] - mu) / sd).reshape(len(tr), -1), y[tr])
                        joblib.dump(model, saved.with_suffix(".joblib"), compress=3)

                    def predict(Z, model=model, mu=mu, sd=sd, multilabel=multilabel):
                        p = model.predict_proba(((Z - mu) / sd).reshape(len(Z), -1))
                        if not multilabel or isinstance(p, np.ndarray):
                            return p
                        # one output per label; a label never positive in training has a single column
                        return np.stack([q[:, list(c).index(1)] if 1 in c else np.zeros(len(q))
                                         for q, c in zip(p, model.classes_, strict=True)], 1)
                else:
                    import tensorflow as tf

                    if saved.with_suffix(".h5").exists():
                        model = tf.keras.models.load_model(saved.with_suffix(".h5"))
                    else:
                        model = keras_model(clf_name, X.shape[1], X.shape[2], n_out, multilabel, seed=fi)
                        stop = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
                        model.fit((X[tr] - mu) / sd, y[tr], epochs=args.epochs, batch_size=64, validation_split=0.1,
                                  callbacks=[stop], verbose=0)
                        model.save(saved.with_suffix(".h5"))

                    def predict(Z, model=model, mu=mu, sd=sd):
                        return model.predict((Z - mu) / sd, verbose=0)
                for name, Z in [("original", X[te])] + [(t, recs[k][te]) for k, t in enumerate(techs)]:
                    proba = predict(Z)
                    per[name].append(scores(y[te], proba, multilabel))
                    if per_clip:
                        uc, inv = np.unique(clips[te], return_inverse=True)
                        mean_p = np.zeros((len(uc), proba.shape[1]))
                        np.add.at(mean_p, inv, proba)
                        clip_y = np.zeros(len(uc), int)
                        clip_y[inv] = y[te]
                        per[name][-1]["clip_accuracy"] = float((mean_p.argmax(1) == clip_y).mean())
                    if not multilabel:
                        correct[name][te] = proba.argmax(1) == y[te]
            entry = {}
            for name, lst in per.items():
                entry[name] = {m: float(np.mean([s[m] for s in lst])) for m in lst[0]}
                if not multilabel and name != "original":
                    diff, ci = group_diff_ci(correct["original"], correct[name], groups, rng)
                    entry[name] |= {"group_acc_diff": diff, "ci95": ci, "equivalent_0.01": bool(-0.01 <= ci[0] and ci[1] <= 0.01)}
            if only:            # keep the published entries; the added ones carry the refit they are compared with
                refit = entry.pop("original")
                res[clf_name] = done[clf_name] | {"original_refit": refit} | {
                    t: entry[t] | {"ref_original": refit} for t in techs}
            else:
                res[clf_name] = entry
            summary[f"{ds}_L{length}"] = res | {"partial": True}
            path.write_text(json.dumps(summary, indent=1) + "\n")
        res["wall_s"] = done.get("wall_s", 0.0) + time.perf_counter() - t0 if only else time.perf_counter() - t0
        res.pop("partial", None)
        summary[f"{ds}_L{length}"] = res
        path.write_text(json.dumps(summary, indent=1) + "\n")
        key = "macro_f1" if multilabel else "accuracy"
        print(f"{ds:13} L{length:<4} {len(X)} win x {X.shape[2]} ch, {res['groups']} groups, {res['wall_s']:.0f} s | {key} {'/'.join(clfs)}: "
              + "  ".join(f"{n} " + "/".join(f"{res[c][n][key]:.3f}" for c in clfs) for n in ("original",) + techs),
              flush=True)
    write_md(summary, out)


def write_md(summary, out=OUT):
    from scripts._benchmark import LABELS

    rows = ["# M7 classification: classifiers trained on original windows, tested on reconstructions", "",
            "Accuracy for multiclass data, macro-F1 for multilabel PTB-XL. Grouped 3-fold, 8-bit images. "
            "Mark ✓: per-group accuracy difference within ±0.01 (95% cluster bootstrap).", ""]
    small = [k for k, r in summary.items() if not k.startswith("_") and r["windows"] < MIN_WINDOWS]
    if small:
        rows += [f"Not reported (fewer than {MIN_WINDOWS} windows): " + ", ".join(small) + ".", ""]
    lengths = sorted({r["length"] for k, r in summary.items() if not k.startswith("_")})
    for length, clf in [(n, c) for n in lengths for c in CLASSIFIERS]:
        entries = {k: r for k, r in summary.items() if not k.startswith("_") and r["length"] == length and clf in r
                   and r["windows"] >= MIN_WINDOWS}
        if not entries:
            continue
        rows += [f"## L = {length} · {clf.upper()}", "", "| Dataset | Original | " + " | ".join(LABELS[t] for t in TECHS) + " |",
                 "|---|---|" + "---|" * len(TECHS)]
        for r in entries.values():
            key = "macro_f1" if r["multilabel"] else "accuracy"
            cells = [f"{r[clf][t][key]:.3f}" + (" ✓" if r[clf][t].get("equivalent_0.01") else "") if t in r[clf] else "—"
                     for t in TECHS]
            rows.append(f"| {r['dataset']} | {r[clf]['original'][key]:.3f} | " + " | ".join(cells) + " |")
        rows.append("")
    (out / "summary.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    main()
