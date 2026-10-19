"""
==============================================================
  STEP 1 (deep learning, optional) - Heart chambers with TotalSegmentator
  UTM-Claude Research Project | rohayanti@utm.my

  Deep-learning labels for the same ids as segment_heart.py (LABELS):
    heartchambers_highres  (licensed; free for non-commercial use)
        1 LV, 2 LA, 3 aorta, 4 myocardium, 5 RA, 6 RV, 7 pulmonary artery
    total                  (open)
        8 SVC, and the aorta when the chambers task is not available
  Without a licence only the open task runs, so the result has the aorta
  and SVC but no chambers.

  TotalSegmentator needs Python >= 3.10 and PyTorch, so it lives in its
  own environment and is called as a command. This script runs in the
  tested 3.9 environment and reads back the NIfTI files it writes.
  The CT goes in at native resolution (the chambers model does not
  resample). The labels come back onto the 1 mm grid of segment_heart.py.

  Both tasks label the whole aorta in the field of view. Like the rules,
  only the ascending aorta (with the root) is kept: the part within
  AORTA_TOUCH_MM of the LV, or else the most anterior part.
  The open task's roi_subset shortcut is not used: on this small cardiac
  field of view its coarse crop missed the SVC.

  Needs segment_heart.py first (grid and rule labels for the comparison).
  Usage:
    python segment_heart_dl.py [--series "CT CCTA DL"] [--totalseg PATH] [--device gpu|cpu]
  TotalSegmentator is found via --totalseg, the HEART3D_TOTALSEG environment
  variable, ~/venvs/heart3d-dl, then PATH (see README_3D.md). The project's own
  GPU environment comes before PATH, where a CPU-only copy may sit.

  Output (output/3d_models):
    heart_labels_dl.nii.gz   DL labels on the heart_labels.nii.gz grid
    dl_segmentation.json     volumes, Dice vs the rules, tasks, versions
  Output (output/visualizations):
    dl_vs_rules_overlay.png
==============================================================
"""
import argparse, json, os, shutil, subprocess, tempfile, time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

import heart_io as hio
from segment_heart import LABELS

CHAMBERS_TASK = "heartchambers_highres"
TOTAL_TASK = "total"
# project label id -> (TotalSegmentator task, class name)
DL_CLASSES = {
    1: (CHAMBERS_TASK, "heart_ventricle_left"),
    2: (CHAMBERS_TASK, "heart_atrium_left"),
    3: (CHAMBERS_TASK, "aorta"),
    4: (CHAMBERS_TASK, "heart_myocardium"),
    5: (CHAMBERS_TASK, "heart_atrium_right"),
    6: (CHAMBERS_TASK, "heart_ventricle_right"),
    7: (CHAMBERS_TASK, "pulmonary_artery"),
    8: (TOTAL_TASK, "superior_vena_cava"),
}
AORTA_TOUCH_MM = 3.0      # ascending aorta = aortic components this close to the LV
AORTA_MIN_SHARE = 0.1     # fallback (no LV): ignore aortic specks below this share of the aorta
DEFAULT_ENV = Path.home() / "venvs" / "heart3d-dl"


# ── TotalSegmentator (separate environment) ───────────────────────────────
def find_totalseg(path=None):
    """TotalSegmentator executable: argument, HEART3D_TOTALSEG, ~/venvs/heart3d-dl, then PATH."""
    exe, bindir = ("TotalSegmentator.exe", "Scripts") if os.name == "nt" else ("TotalSegmentator", "bin")
    for c in (path, os.environ.get("HEART3D_TOTALSEG"), DEFAULT_ENV / bindir / exe, shutil.which("TotalSegmentator")):
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError("TotalSegmentator not found. Install it in a Python >= 3.10 environment, then pass "
                            "--totalseg or set HEART3D_TOTALSEG to its executable (see README_3D.md).")


def _run(cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))


def env_python(ts):
    """The Python of TotalSegmentator's environment: next to the executable (a venv's Scripts / bin), or one folder
    up (a system Python's Scripts folder)."""
    name = "python.exe" if os.name == "nt" else "python"
    for c in (ts.with_name(name), ts.parent.parent / name):
        if c.is_file():
            return c
    raise FileNotFoundError(f"no Python found for {ts}")


def _ask(ts, code, *args):
    """Run a snippet with the TotalSegmentator environment's Python; returns its last output line."""
    r = _run([env_python(ts), "-c", code, *args])
    if r.returncode != 0:
        raise RuntimeError(f"TotalSegmentator environment check failed:\n{r.stderr[-1000:]}")
    return r.stdout.strip().splitlines()[-1]


def class_map(ts, task):
    """{class name: label value} of a task's multilabel output."""
    code = "import json,sys; from totalsegmentator.map_to_binary import class_map as c; print(json.dumps(c[sys.argv[1]]))"
    return {name: int(v) for v, name in json.loads(_ask(ts, code, task)).items()}


def has_licence(ts):
    """True when a licence number is set (checked offline; the number itself is never read here)."""
    return _ask(ts, "from totalsegmentator.config import has_valid_license_offline as h; print(h()[0])") == "yes"


def ts_version(ts):
    return _ask(ts, "import importlib.metadata as m; print(m.version('TotalSegmentator'))")


def run_task(ts, ct_path, out_path, task, device="gpu"):
    """One TotalSegmentator run -> one multilabel NIfTI (values as in class_map)."""
    r = _run([ts, "-i", ct_path, "-o", out_path, "--ml", "--task", task, "--device", device, "--nr_thr_saving", 1])
    if r.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError(f"TotalSegmentator task '{task}' failed:\n{(r.stdout + r.stderr)[-1500:]}")


# ── labels ────────────────────────────────────────────────────────────────
def to_grid(path, ref):
    """Label image resampled onto the reference grid (label-aware Gaussian, no new labels)."""
    return sitk.GetArrayFromImage(sitk.Resample(sitk.ReadImage(str(path)), ref, sitk.Transform(),
                                                sitk.sitkLabelGaussian, 0))


def ascending_aorta(aorta, lv, spacing_zyx):
    """Ascending aorta + root: aortic components within AORTA_TOUCH_MM of the LV.

    Without an LV (or if none is close), the most anterior component (smallest y
    in LPS index order) among those holding >= AORTA_MIN_SHARE of the aorta.
    """
    lab, n = ndi.label(aorta)
    if n <= 1:
        return aorta
    if lv is not None and lv.any():
        near = ndi.distance_transform_edt(~lv, sampling=spacing_zyx) <= AORTA_TOUCH_MM
        keep = np.unique(lab[near & aorta])
        keep = keep[keep > 0]
        if len(keep):
            return np.isin(lab, keep)
    ids = np.arange(1, n + 1)
    sizes = ndi.sum(aorta, lab, ids)
    big = ids[sizes >= AORTA_MIN_SHARE * sizes.sum()]
    return lab == min(big, key=lambda i: ndi.center_of_mass(lab == i)[1])


def project_labels(maps, spacing_zyx):
    """Label volume in the ids of segment_heart.LABELS.

    maps: {task: (label array on the 1 mm grid, {class name: value})}. Chamber
    classes come from one multilabel image, so they never overlap; the SVC
    (open task) only fills voxels no chamber claimed. The aorta comes from the
    chambers task when it ran, else from the open task. Returns (labels, sources).
    """
    shape = next(iter(maps.values()))[0].shape
    labels = np.zeros(shape, np.uint8)
    sources = {}

    def mask(task, name):
        arr, cmap = maps[task]
        return arr == cmap[name]

    for lid, (task, name) in DL_CLASSES.items():
        if lid == 3 and task not in maps and TOTAL_TASK in maps:
            task = TOTAL_TASK                      # aorta fallback
        if task not in maps:
            continue
        m = mask(task, name)
        if lid == 3:
            lv = mask(*DL_CLASSES[1]) if DL_CLASSES[1][0] in maps else None
            m = ascending_aorta(m, lv, spacing_zyx)
        labels[m & (labels == 0)] = lid
        sources[LABELS[lid][0]] = f"{task}:{name}"
    return labels, sources


# ── comparison + QC ───────────────────────────────────────────────────────
def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else None


def compare(rules, dl, vox_ml):
    """Volumes (mL) and Dice per label, for the labels present in the DL result."""
    rows = {}
    for lid, (name, _) in LABELS.items():
        r, d = rules == lid, dl == lid
        rows[name] = {"rules_ml": round(float(r.sum() * vox_ml), 1), "dl_ml": round(float(d.sum() * vox_ml), 1),
                      "dice": round(dice(r, d), 3) if d.any() else None}
    return rows


def save_overlay(ct, rules, dl, path):
    """Rules (top) vs deep learning (bottom): 3 axial levels + coronal, cropped to the heart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    both = (rules > 0) | (dl > 0)
    zz, yy, xx = [np.where(both.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
    pad = 12
    zs = slice(max(zz[0] - pad, 0), zz[-1] + pad)
    ys, xs = slice(max(yy[0] - pad, 0), yy[-1] + pad), slice(max(xx[0] - pad, 0), xx[-1] + pad)
    levels = np.linspace(zz[0], zz[-1], 5)[1:-1].astype(int)
    yc = int(np.mean(np.where(both)[1]))
    colours = {lid: np.array(LABELS[lid][1][:3]) for lid in LABELS}

    def paint(img, lab):
        rgb = np.stack([np.clip((img + 200) / 800, 0, 1)] * 3, -1)
        for lid, c in colours.items():
            m = lab == lid
            rgb[m] = rgb[m] * 0.45 + c * 0.55
        return rgb

    fig, axs = plt.subplots(2, 4, figsize=(17, 9), facecolor="#111")
    for row, (lab, title) in enumerate([(rules, "Rules (segment_heart.py)"), (dl, "Deep learning (TotalSegmentator)")]):
        views = [(f"axial z={z}", ct[z][ys, xs], lab[z][ys, xs]) for z in levels]
        views.append(("coronal", ct[zs, yc, xs][::-1], lab[zs, yc, xs][::-1]))
        for ax, (name, img, lb) in zip(axs[row], views):
            ax.imshow(paint(img, lb))
            ax.set_title(f"{title}\n{name}", color="w", fontsize=9)
            ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABELS[i][1][:3]) for i in sorted(LABELS)]
    fig.legend(handles, [LABELS[i][0] for i in sorted(LABELS)], loc="lower center", ncol=8,
               facecolor="#222", labelcolor="w", fontsize=8)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(path, dpi=70, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────
def main(series=hio.DEFAULT_SERIES, totalseg=None, device="gpu", log=print):
    t0 = time.time()
    ref_path = os.path.join(hio.OUT_MODELS, "heart_ct_1mm.nii.gz")
    rules_path = os.path.join(hio.OUT_MODELS, "heart_labels.nii.gz")
    if not (os.path.exists(ref_path) and os.path.exists(rules_path)):
        raise FileNotFoundError("run segment_heart.py first (needs heart_ct_1mm.nii.gz and heart_labels.nii.gz)")
    ts = find_totalseg(totalseg)
    version, licence = ts_version(ts), has_licence(ts)
    tasks = ([CHAMBERS_TASK] if licence else []) + [TOTAL_TASK]
    log(f"[1/4] TotalSegmentator {version} ({ts.parent}); tasks: {', '.join(tasks)}")
    if not licence:
        log(f"      no licence set: '{CHAMBERS_TASK}' skipped, so no chambers (set one with totalseg_set_license)")
    ref = sitk.ReadImage(ref_path)
    spacing_zyx = ref.GetSpacing()[::-1]
    with tempfile.TemporaryDirectory(prefix="heart_dl_") as tmp:   # native CT + raw outputs, deleted afterwards
        log(f"[2/4] Exporting {series} at native resolution ...")
        ct = hio.load_series(os.path.join(hio.DATASET, series))
        ct = sitk.Clamp(ct, sitk.sitkFloat32, -1024.0, 32767.0)     # outside-FOV padding (-3024) -> air
        ct_path = os.path.join(tmp, "ct.nii.gz")
        sitk.WriteImage(ct, ct_path)
        log(f"      {ct.GetSize()} @ {tuple(round(s, 3) for s in ct.GetSpacing())} mm")
        log("[3/4] Running TotalSegmentator ...")
        maps = {}
        for task in tasks:
            t = time.time()
            out = os.path.join(tmp, f"{task}.nii.gz")
            run_task(ts, ct_path, out, task, device)
            maps[task] = (to_grid(out, ref), class_map(ts, task))
            log(f"      {task}: {time.time() - t:.0f}s")
    log("[4/4] Mapping labels and comparing with the rules ...")
    DL_LABELS = os.path.join(hio.OUT_MODELS, "heart_labels_dl.nii.gz")      # looked up now: OUT_MODELS may be redirected
    DL_STATS = os.path.join(hio.OUT_MODELS, "dl_segmentation.json")
    DL_OVERLAY = os.path.join(hio.OUT_VIS, "dl_vs_rules_overlay.png")
    labels, sources = project_labels(maps, spacing_zyx)
    lab_img = sitk.GetImageFromArray(labels)
    lab_img.CopyInformation(ref)
    sitk.WriteImage(lab_img, DL_LABELS)
    rules = sitk.GetArrayFromImage(sitk.ReadImage(rules_path))
    vox_ml = float(np.prod(ref.GetSpacing())) / 1000.0
    rows = compare(rules, labels, vox_ml)
    stats = {"series": series, "totalsegmentator": version, "device": device, "tasks": tasks,
             "chambers": licence, "sources": sources,
             "labels": {n: r["dl_ml"] for n, r in rows.items() if r["dice"] is not None},
             "whole_heart_ml": round(float((labels > 0).sum() * vox_ml), 1),
             "vs_rules": rows,
             "whole_heart_dice_vs_rules": round(dice(rules > 0, labels > 0), 3) if licence else None,
             "runtime_s": round(time.time() - t0)}
    with open(DL_STATS, "w") as fh:
        json.dump(stats, fh, indent=2)
    save_overlay(sitk.GetArrayFromImage(ref), rules, labels, DL_OVERLAY)
    for n, r in rows.items():
        if r["dice"] is not None:
            log(f"      {n:20s} DL {r['dl_ml']:6.1f} mL | rules {r['rules_ml']:6.1f} mL | Dice {r['dice']:.3f}")
    log(f"Done in {stats['runtime_s']}s -> {DL_LABELS}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=hio.DEFAULT_SERIES)
    ap.add_argument("--totalseg", help="TotalSegmentator executable (default: HEART3D_TOTALSEG, PATH, ~/venvs/heart3d-dl)")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    a = ap.parse_args()
    main(a.series, a.totalseg, a.device)
