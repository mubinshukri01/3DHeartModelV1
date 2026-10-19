"""
==============================================================
  EAT benchmark on the Rodriguez cardiac-fat dataset
  UTM-Claude Research Project | rohayanti@utm.my

  Scores eat_segment.py against the expert EAT masks of 18 patients
  (TJes, a single slice, is left out). The classifier is evaluated
  leave-one-patient-out: each patient is segmented by a model trained
  on the other 17, so every number is on unseen data.
  Dice is computed per patient over all slices on the annotation grid.
  Fat is the annotators' own fat pixels, so it measures the EAT
  decision. Pixels on the drawn pericardium line are ignored.

  Usage (from scripts/, ~45 min on a laptop CPU):
    python eat_benchmark.py                # classifier, LOPO CV
    python eat_benchmark.py --rules-only   # training-free baseline (~15 min)
    python eat_benchmark.py --save-model   # LOPO + final model on all 18
                                           #   -> models/eat_classifier.joblib

  Output (output/eat_benchmark/):
    results_<method>.csv, summary_<method>.json,
    agreement_<method>.png (volume scatter + Bland-Altman),
    examples_<method>.png (best / median / worst patient)
==============================================================
"""
import argparse, csv, json, os, time

import numpy as np

import eat_segment as es
import heart_io as hio
import rodriguez as rz

OUT = os.path.join(hio.PROJECT, "output", "eat_benchmark")
SAMPLES_PER_PATIENT = 60000    # training voxels drawn per patient (balanced by patient, not by size)
CLASSIFIER_PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                         l2_regularization=1.0, random_state=0)


def prepare(pid, log=print):
    """Load one patient and compute candidate features on the annotation grid."""
    t = time.time(); P = rz.load_patient(pid)
    X, idx, _ = es.features(P["hu"], P["px"], P["z_mm"], fat=P["fat"], fat_range=rz.FAT_RANGE)
    y = (P["eat"] & ~P["ignore"])[tuple(idx.T)]
    log(f"  {pid}: {len(y):,} candidate fat voxels, {y.mean():.0%} EAT ({time.time() - t:.0f}s)")
    return dict(X=X, y=y, idx=idx, eat=P["eat"], ignore=P["ignore"], hu=P["hu"], px=P["px"],
                vox_ml=P["px"] ** 2 * P["z_mm"] / 1000.0)


def score_patient(D, pred_idx):
    pred = np.zeros(D["eat"].shape, bool); pred[tuple(pred_idx.T)] = True
    pred &= ~D["ignore"]; gt = D["eat"] & ~D["ignore"]; tp = (pred & gt).sum()
    return dict(dice=2 * tp / max(pred.sum() + gt.sum(), 1), precision=tp / max(pred.sum(), 1),
                recall=tp / max(gt.sum(), 1), gt_ml=gt.sum() * D["vox_ml"], pred_ml=pred.sum() * D["vox_ml"]), pred


def fit(Xs, ys):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**CLASSIFIER_PARAMS).fit(np.concatenate(Xs), np.concatenate(ys))


def summarise(rows):
    d = np.array([r["dice"] for r in rows]); g = np.array([r["gt_ml"] for r in rows]); p = np.array([r["pred_ml"] for r in rows])
    diff = p - g
    return dict(patients=len(rows), dice_mean=float(d.mean()), dice_sd=float(d.std(ddof=1)), dice_median=float(np.median(d)),
                dice_min=float(d.min()), dice_max=float(d.max()),
                precision_mean=float(np.mean([r["precision"] for r in rows])), recall_mean=float(np.mean([r["recall"] for r in rows])),
                volume_r=float(np.corrcoef(g, p)[0, 1]), volume_bias_ml=float(diff.mean()),
                volume_loa_ml=[float(diff.mean() - 1.96 * diff.std(ddof=1)), float(diff.mean() + 1.96 * diff.std(ddof=1))],
                volume_mae_ml=float(np.abs(diff).mean()), gt_volume_ml_mean=float(g.mean()))


def save_figures(rows, preds, data, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.array([r["gt_ml"] for r in rows]); p = np.array([r["pred_ml"] for r in rows]); s = summarise(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    lim = [0, max(g.max(), p.max()) * 1.1]
    a1.plot(lim, lim, color="#999", ls="--", lw=1); a1.scatter(g, p, color="#c0392b", zorder=3)
    for r in rows: a1.annotate(r["pid"], (r["gt_ml"], r["pred_ml"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a1.set(xlim=lim, ylim=lim, xlabel="Expert EAT volume (mL)", ylabel="Predicted EAT volume (mL)",
           title=f"Volume agreement (r = {s['volume_r']:.2f})")
    m = (g + p) / 2; d = p - g
    a2.scatter(m, d, color="#2980b9", zorder=3); a2.axhline(s["volume_bias_ml"], color="#333")
    for v in s["volume_loa_ml"]: a2.axhline(v, color="#999", ls="--")
    a2.set(xlabel="Mean of expert and predicted (mL)", ylabel="Predicted - expert (mL)",
           title=f"Bland-Altman: bias {s['volume_bias_ml']:+.1f} mL, 95% LoA [{s['volume_loa_ml'][0]:.0f}, {s['volume_loa_ml'][1]:.0f}]")
    fig.suptitle(f"EAT on Rodriguez ({method}, leave-one-patient-out): Dice {s['dice_mean']:.3f} ± {s['dice_sd']:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"agreement_{method}.png"), dpi=110); plt.close(fig)

    order = sorted(rows, key=lambda r: r["dice"])
    picks = [("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="#111")
    for ax, (tag, r) in zip(axs, picks):
        D = data[r["pid"]]; pred = preds[r["pid"]]; gt = D["eat"] & ~D["ignore"]
        k = int(np.argmax(gt.sum((1, 2))))                       # slice with the most EAT
        img = np.stack([np.clip((D["hu"][k] + 160) / 400, 0, 1)] * 3, -1)
        img[gt[k] & pred[k]] = (1.0, 0.85, 0.0); img[gt[k] & ~pred[k]] = (1.0, 0.1, 0.1); img[pred[k] & ~gt[k]] = (0.0, 0.85, 1.0)
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"{tag}: {r['pid']}  Dice {r['dice']:.2f}", color="w")
    fig.text(0.5, 0.02, "yellow = correct EAT    red = missed    cyan = false EAT", color="w", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(OUT, f"examples_{method}.png"), dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(rules_only=False, save_model=False, log=print):
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    method = "rules" if rules_only else "classifier"
    pids = rz.patient_ids()
    log(f"[1/3] Features for {len(pids)} patients ...")
    data = {p: prepare(p, log) for p in pids}
    rng = np.random.default_rng(0)
    samples = {}
    for p, D in data.items():
        i = rng.choice(len(D["y"]), min(SAMPLES_PER_PATIENT, len(D["y"])), replace=False)
        samples[p] = (D["X"][i], D["y"][i])
    log(f"[2/3] {'Rules-only' if rules_only else 'Leave-one-patient-out'} segmentation ...")
    rows, preds = [], {}
    for p, D in data.items():
        if rules_only:
            score = (D["X"][:, es.FEATURES.index("dist_outline")] <= es.RULES_MARGIN_MM).astype(np.float32); thr = 0.5
        else:
            clf = fit([samples[q][0] for q in pids if q != p], [samples[q][1] for q in pids if q != p])
            score = es.smooth_scores(D["eat"].shape, D["idx"], clf.predict_proba(D["X"])[:, 1], D["px"])
            thr = es.THRESHOLD
        r, preds[p] = score_patient(D, D["idx"][score >= thr]); r["pid"] = p; rows.append(r)
        log(f"  {p}: Dice {r['dice']:.3f}  expert {r['gt_ml']:6.1f} mL  predicted {r['pred_ml']:6.1f} mL")
    s = summarise(rows)
    log(f"[3/3] Dice {s['dice_mean']:.3f} ± {s['dice_sd']:.3f} (median {s['dice_median']:.3f}, range "
        f"{s['dice_min']:.3f}-{s['dice_max']:.3f}); volume r = {s['volume_r']:.3f}, bias {s['volume_bias_ml']:+.1f} mL")
    with open(os.path.join(OUT, f"results_{method}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, ["pid", "dice", "precision", "recall", "gt_ml", "pred_ml"]); w.writeheader()
        for r in rows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    s.update(method=method, fat_range_hu=list(rz.FAT_RANGE), threshold=None if rules_only else es.THRESHOLD,
             smooth_mm=None if rules_only else es.SMOOTH_MM, classifier_params=None if rules_only else CLASSIFIER_PARAMS,
             runtime_s=round(time.time() - t0))
    with open(os.path.join(OUT, f"summary_{method}.json"), "w") as fh:
        json.dump(s, fh, indent=2)
    save_figures(rows, preds, data, method)
    if save_model and not rules_only:
        import joblib, sklearn
        clf = fit([samples[q][0] for q in pids], [samples[q][1] for q in pids])
        os.makedirs(os.path.dirname(es.MODEL_PATH), exist_ok=True)
        joblib.dump(dict(classifier=clf, features=es.FEATURES, threshold=es.THRESHOLD, smooth_mm=es.SMOOTH_MM,
                         trained_on=pids, sklearn_version=sklearn.__version__, lopo_summary=s), es.MODEL_PATH, compress=3)
        log(f"Saved model trained on all {len(pids)} patients -> {es.MODEL_PATH}")
    log(f"Done in {time.time() - t0:.0f}s -> {OUT}")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-only", action="store_true", help="training-free baseline, no classifier")
    ap.add_argument("--save-model", action="store_true", help="also fit the final model on all patients")
    a = ap.parse_args()
    main(a.rules_only, a.save_model)
