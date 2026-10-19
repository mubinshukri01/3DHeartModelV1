"""
==============================================================
  STEP 1d (optional) - Great vessels with TotalSegmentator (open task)
  UTM-Claude Research Project | rohayanti@utm.my

  The rule-based labels (segment_heart.py) have the ascending aorta, the
  pulmonary artery and an approximate SVC. TotalSegmentator's open 'total'
  task (no licence needed) adds what the rules lack, on the same 1 mm grid:
    * SVC: replaces the rule SVC, which lay in the right atrial
      appendage (step 10 review); that region goes to the RA
    * IVC, pulmonary veins and the descending aorta (new structures)
    * the left atrial appendage (joins the LA) and the aortic root (joins
      the aorta), where the rules left the catch-all 'Myocardium' label
      (it covered the front of the aortic root as a thin sheet)
  Each structure may only take over the rule labels in REPLACES, so the
  measured chambers (LV, LA body, RV) keep all their voxels. Vessels end
  where the scan ends (140 mm), with flat caps; the aortic arch is above
  the scan and is not added.

  Usage (after segment_heart.py):
    python vessels.py [--totalseg PATH] [--device gpu|cpu]
  Output (output/3d_models): vessels_dl.nii.gz (the TotalSegmentator classes
  of VESSEL_CLASSES on the heart_labels grid), vessels_dl.json.
  build_heart_model.py merges them into the 3D model (merge()) when present.
==============================================================
"""
import argparse, json, os, tempfile, time

import numpy as np
import SimpleITK as sitk

import heart_io as hio
import segment_heart_dl as dl

# values in vessels_dl.nii.gz -> TotalSegmentator 'total' classes
VESSEL_CLASSES = {1: "superior_vena_cava", 2: "inferior_vena_cava", 3: "pulmonary_vein", 4: "aorta",
                  5: "atrial_appendage_left", 6: "heart"}
# new structures of the 3D model (ids continue segment_heart.LABELS)
VESSEL_LABELS = {9: ("Inferior vena cava", (0.18, 0.30, 0.66, 1.0)),
                 10: ("Pulmonary veins", (0.80, 0.30, 0.40, 1.0)),
                 11: ("Descending aorta", (0.98, 0.75, 0.20, 1.0))}
VESSEL_SLUGS = {9: "ivc", 10: "pv", 11: "desc_aorta"}
# model label -> rule labels it may take over (0 background, 4 catch-all 'Myocardium', 5 RA, 7 PA, 8 rule SVC)
REPLACES = {8: {0, 4, 5, 7, 8}, 9: {0, 4, 5}, 10: {0, 4}, 11: {0, 4}, 2: {0, 4}, 3: {0, 4}}
# with the deep-learning chambers (segment_heart_dl.py) every label is anatomical (4 is the true LV wall, 8 already
# TotalSegmentator's SVC), so the vessels only fill background; the venae cavae also their junction with the RA
REPLACES_DL = {8: {0, 5, 8}, 9: {0, 5}, 10: {0}, 11: {0}, 2: {0}, 3: {0}}


def vessels_file():
    return os.path.join(hio.OUT_MODELS, "vessels_dl.nii.gz")


def stats_file():
    return os.path.join(hio.OUT_MODELS, "vessels_dl.json")


def merge(rules, ves, spacing_zyx, deep=False):
    """Model labels: the chamber labels with the TotalSegmentator vessels merged in (see the module docstring).

    rules: chamber labels (segment_heart.py, or segment_heart_dl.py when deep); ves: vessels_dl values
    (VESSEL_CLASSES) on the same grid.
    """
    out = rules.copy()
    replaces = REPLACES_DL if deep else REPLACES
    if not deep:
        old_svc = rules == 8
        out[old_svc & (ves == 6)] = 5        # rule 'SVC' inside the heart: right atrial appendage -> RA
        out[old_svc & (ves != 6)] = 0
    aorta = ves == 4
    asc = dl.ascending_aorta(aorta, rules == 1, spacing_zyx) if aorta.any() else aorta
    for lid, m in ((8, ves == 1), (9, ves == 2), (10, ves == 3), (11, aorta & ~asc), (2, ves == 5), (3, asc)):
        out[m & np.isin(out, list(replaces[lid]))] = lid
    return out


def load(lab_img, log=print):
    """vessels_dl array on the heart_labels grid, or None (not run, or made on another grid)."""
    if not os.path.exists(vessels_file()):
        return None
    img = sitk.ReadImage(vessels_file())
    if img.GetSize() != lab_img.GetSize() or not (np.allclose(img.GetSpacing(), lab_img.GetSpacing())
                                                  and np.allclose(img.GetOrigin(), lab_img.GetOrigin())):
        log("  vessels_dl.nii.gz is on another grid (re-run vessels.py) - skipping the vessels")
        return None
    return sitk.GetArrayFromImage(img)


def main(series=hio.DEFAULT_SERIES, totalseg=None, device="gpu", log=print):
    t0 = time.time()
    ref_path = os.path.join(hio.OUT_MODELS, "heart_ct_1mm.nii.gz")
    if not os.path.exists(ref_path):
        raise FileNotFoundError("run segment_heart.py first (needs heart_ct_1mm.nii.gz)")
    ts = dl.find_totalseg(totalseg)
    version = dl.ts_version(ts)
    log(f"[1/3] TotalSegmentator {version}: exporting {series} at native resolution ...")
    ref = sitk.ReadImage(ref_path)
    with tempfile.TemporaryDirectory(prefix="heart_vessels_") as tmp:     # native CT + raw output, deleted after
        ct = sitk.Clamp(hio.load_series(os.path.join(hio.DATASET, series)), sitk.sitkFloat32, -1024.0, 32767.0)
        ct_path, raw = os.path.join(tmp, "ct.nii.gz"), os.path.join(tmp, "total.nii.gz")
        sitk.WriteImage(ct, ct_path)
        log("[2/3] Running the open 'total' task ...")
        dl.run_task(ts, ct_path, raw, "total", device)
        total = dl.to_grid(raw, ref)
    cmap = dl.class_map(ts, "total")
    ves = np.zeros(total.shape, np.uint8)
    for v, name in VESSEL_CLASSES.items():
        ves[total == cmap[name]] = v
    log("[3/3] Saving ...")
    out = sitk.GetImageFromArray(ves)
    out.CopyInformation(ref)
    sitk.WriteImage(out, vessels_file())
    vox_ml = float(np.prod(ref.GetSpacing())) / 1000.0
    stats = {"series": series, "totalsegmentator": version, "task": "total", "device": device,
             "classes": VESSEL_CLASSES,
             "volumes_ml": {name: round(float((ves == v).sum() * vox_ml), 1) for v, name in VESSEL_CLASSES.items()},
             "runtime_s": round(time.time() - t0)}
    with open(stats_file(), "w") as fh:
        json.dump(stats, fh, indent=2)
    log(f"Done in {stats['runtime_s']}s -> {vessels_file()}: {stats['volumes_ml']}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=hio.DEFAULT_SERIES)
    ap.add_argument("--totalseg", help="TotalSegmentator executable (default: HEART3D_TOTALSEG, PATH, ~/venvs/heart3d-dl)")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    a = ap.parse_args()
    main(a.series, a.totalseg, a.device)
