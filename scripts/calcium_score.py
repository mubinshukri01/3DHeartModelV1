"""
==============================================================
  STEP 1c - Coronary artery calcium: Agatston score (semi-automatic)
  UTM-Claude Research Project | rohayanti@utm.my

  Standard Agatston scoring on the non-contrast CT CALCIUM SCORE series:
    lesion  = 2D connected region >= 130 HU with area >= 1 mm^2 (per slice)
    score   = sum of lesion area (mm^2) x density weight of its peak HU
              (1: 130-199, 2: 200-299, 3: 300-399, 4: >= 400),
              x slice spacing / 3 mm (Agatston's reference slices)
    volume  = total lesion volume (mm^3)
  Only coronary calcium counts. Like the scanner's own scoring software,
  this is semi-automatic:
    1  every lesion in the coronary territory is SUGGESTED automatically:
       inside the heart zone (eat_segment.py outline) and within
       EAT_NEAR_MM of epicardial fat, where the coronary arteries run.
       Bone, valve and blood-pool spots are not suggested.
    2  a reader confirms or rejects each suggestion in the app
       ("Coronary calcium" in the 3D tab); calcium_review.json keeps it.
  Tiny faint spots at tissue edges meet the Agatston criteria too, so
  the automatic score is an upper estimate until reviewed.

  Needs eat_patient.py first (eat_cac.nii.gz).
  Usage:  python calcium_score.py [--series "CT CALCIUM SCORE"]
  Output: output/3d_models/calcium_stats.json,
          output/visualizations/calcium_overlay.png, calcium/lesion_<id>.png
==============================================================
"""
import argparse, glob, json, os, re, time

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

import eat_segment as es
import heart_io as hio

CAC_SERIES = "CT CALCIUM SCORE"
CALCIUM_HU = 130            # Agatston threshold
MIN_AREA_MM2 = 1.0          # smallest lesion per slice
MAX_AREA_MM2 = 200.0        # bigger bright regions are bone (sternum, vertebra), not plaque
AGATSTON_REF_MM = 3.0       # Agatston's slice thickness; other spacings are scaled to it
ZONE_MARGIN_MM = 5.0        # heart zone = pericardial outline widened by this
EAT_NEAR_MM = 3.0           # coronary lesions lie within this distance of epicardial fat
THUMB_PX = 64               # review thumbnail crop (pixels)


def lesions_dir():
    return os.path.join(hio.OUT_VIS, "calcium")


def review_path():
    return os.path.join(hio.OUT_MODELS, "calcium_review.json")


def density_weight(peak_hu):
    """Agatston weight of a lesion from its peak HU (0 below the threshold)."""
    return 0 if peak_hu < CALCIUM_HU else 1 if peak_hu < 200 else 2 if peak_hu < 300 else 3 if peak_hu < 400 else 4


def find_lesions(hu, px, z_mm, eat, zone):
    """All calcium lesions near the heart, grouped in 3D, each marked suggested (coronary territory) or not.

    hu, eat, zone: (Z,Y,X). Returns (lesions, label volume of 3D lesion ids).
    Each lesion: id, slices, peak_hu, area_mm2 (summed over slices), agatston (scaled), volume_mm3,
    suggested, reason, centre (k, y, x), and its per-slice parts.
    """
    near_zone = np.stack([ndi.distance_transform_edt(~z) * px <= ZONE_MARGIN_MM if z.any() else z for z in zone])
    d_eat = np.stack([ndi.distance_transform_edt(~e) * px if e.any() else np.full(e.shape, 1e3) for e in eat])
    calc = hu >= CALCIUM_HU
    keep2d = np.zeros_like(calc)
    parts = []                                               # per-slice lesions that pass the area rule
    for k in range(len(hu)):
        lab, n = ndi.label(calc[k], structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            m = lab == i; area = m.sum() * px * px
            if area < MIN_AREA_MM2 or not (m & near_zone[k]).any():
                continue
            keep2d[k][m] = True
            parts.append(dict(k=k, mask=m, area=area, peak=float(hu[k][m].max()), d_eat=float(d_eat[k][m].min())))
    lab3, n3 = ndi.label(keep2d, structure=np.ones((3, 3, 3)))  # a plaque over several slices = one lesion
    scale = z_mm / AGATSTON_REF_MM
    lesions = []
    for lid in range(1, n3 + 1):
        ps = [p for p in parts if lab3[p["k"]][p["mask"]].max() == lid]
        big = max(p["area"] for p in ps)
        near = min(p["d_eat"] for p in ps)
        if big > MAX_AREA_MM2:
            suggested, reason = False, "bone-sized region"
        elif near > EAT_NEAR_MM:
            suggested, reason = False, f"not next to epicardial fat ({near:.0f} mm)"
        else:
            suggested, reason = True, f"in coronary territory ({near:.1f} mm from EAT)"
        best = max(ps, key=lambda p: p["area"])
        cy, cx = ndi.center_of_mass(best["mask"])
        lesions.append(dict(id=lid, slices=sorted({p["k"] for p in ps}), peak_hu=round(max(p["peak"] for p in ps)),
                            area_mm2=round(sum(p["area"] for p in ps), 2),
                            agatston=round(sum(p["area"] * density_weight(p["peak"]) for p in ps) * scale, 2),
                            volume_mm3=round(sum(p["area"] for p in ps) * z_mm, 1),
                            suggested=suggested, reason=reason, centre=[best["k"], int(cy), int(cx)]))
    return lesions, lab3


def cac_drs(score):
    """Coronary Artery Calcium Data and Reporting System (CAC-DRS) category."""
    return "0 (none)" if score <= 0 else "1 (mild, 1-99)" if score < 100 else "2 (moderate, 100-299)" if score < 300 else "3 (severe, >= 300)"


def reported_agatston(dataset=hio.DATASET):
    """Agatston value from the radiology report (DICOM structured report) if the dataset has one."""
    import pydicom
    for f in glob.glob(os.path.join(dataset, "SR*", "*.dcm")):
        try:
            ds = pydicom.dcmread(f)
        except Exception:
            continue
        texts, stack = [], [ds]
        while stack:
            for el in stack.pop():
                if el.VR == "SQ":
                    stack.extend(el.value)
                elif el.keyword == "TextValue":
                    texts.append(str(el.value))
        m = re.search(r"CALCIUM SCORE\s+([\d.]+)\s*Agatston", "\n".join(texts), re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def score(lesions, confirmed=None):
    """Agatston / volume over suggested lesions, or over the reader-confirmed ids."""
    use = [l for l in lesions if (l["id"] in confirmed if confirmed is not None else l["suggested"])]
    return round(sum(l["agatston"] for l in use), 1), round(sum(l["volume_mm3"] for l in use), 1), len(use)


def save_figures(hu, eat, lab3, lesions, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(lesions_dir(), exist_ok=True)
    for f in glob.glob(os.path.join(lesions_dir(), "lesion_*.png")):
        os.remove(f)
    window = lambda a: np.clip((a + 160) / 400, 0, 1)
    for l in lesions:
        if not l["suggested"]:
            continue
        k, cy, cx = l["centre"]; h = THUMB_PX // 2
        y0, x0 = max(cy - h, 0), max(cx - h, 0)
        sl = (slice(y0, y0 + THUMB_PX), slice(x0, x0 + THUMB_PX))
        img = np.stack([window(hu[k][sl])] * 3, -1)
        img[eat[k][sl]] = img[eat[k][sl]] * 0.55 + np.array([1.0, 0.8, 0.1]) * 0.45
        img[lab3[k][sl] == l["id"]] = (1.0, 0.1, 0.1)
        Image.fromarray((img * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST).save(
            os.path.join(lesions_dir(), f"lesion_{l['id']}.png"))
    sug = [l for l in lesions if l["suggested"]]
    ks = sorted({l["centre"][0] for l in sug})[:4] or [len(hu) // 2]
    fig, axs = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4.6), facecolor="#111", squeeze=False)
    for ax, k in zip(axs[0], ks):
        img = np.stack([window(hu[k])] * 3, -1)
        for l in lesions:
            if k in l["slices"]:
                img[lab3[k] == l["id"]] = (1.0, 0.1, 0.1) if l["suggested"] else (0.2, 0.6, 1.0)
        ax.imshow(img); ax.axis("off"); ax.set_title(f"slice {k}", color="w")
    fig.text(0.5, 0.02, "red = suggested coronary lesion (to review)    blue = >= 130 HU, not coronary territory",
             color="w", ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(path, dpi=80, facecolor=fig.get_facecolor()); plt.close(fig)


def main(series=CAC_SERIES, log=print):
    t0 = time.time()
    eat_path = os.path.join(hio.OUT_MODELS, "eat_cac.nii.gz")
    if not os.path.exists(eat_path):
        raise FileNotFoundError("run eat_patient.py first (needs eat_cac.nii.gz)")
    log(f"[1/3] Loading {series} ...")
    img = hio.load_series(os.path.join(hio.DATASET, series))
    flip = img.GetDirection()[8] > 0                            # work top (slice 0) to bottom, like eat_segment
    hu = np.maximum(sitk.GetArrayFromImage(img), -1024)
    eat = sitk.GetArrayFromImage(sitk.ReadImage(eat_path)) > 0
    if flip:
        hu, eat = hu[::-1].copy(), eat[::-1].copy()
    px, z_mm = img.GetSpacing()[0], img.GetSpacing()[2]
    log("[2/3] Finding calcium lesions ...")
    heart, _ = es.heart_mask(hu, px, z_mm)
    _, zone, _ = es.outlines(heart, px, z_mm)
    lesions, lab3 = find_lesions(hu, px, z_mm, eat, zone)
    auto, auto_vol, n_sug = score(lesions)
    report = reported_agatston()
    log(f"      {len(lesions)} lesions >= {CALCIUM_HU} HU near the heart, {n_sug} suggested coronary: "
        f"automatic Agatston {auto} (volume {auto_vol} mm3)" + (f"; radiology report: {report:g}" if report is not None else ""))
    log("[3/3] Saving stats and review images ...")
    stats = {"series": series, "slice_mm": round(z_mm, 3), "pixel_mm": round(px, 4), "agatston_scale": round(z_mm / AGATSTON_REF_MM, 4),
             "automatic": {"agatston": auto, "volume_mm3": auto_vol, "lesions": n_sug, "cac_drs": cac_drs(auto)},
             "report_agatston": report,
             "criteria": {"threshold_hu": CALCIUM_HU, "min_area_mm2": MIN_AREA_MM2, "max_area_mm2": MAX_AREA_MM2,
                          "zone_margin_mm": ZONE_MARGIN_MM, "eat_near_mm": EAT_NEAR_MM},
             "excluded": {r: sum(1 for l in lesions if not l["suggested"] and l["reason"].startswith(r))
                          for r in ("bone-sized region", "not next to epicardial fat")},
             "lesions": [l for l in lesions if l["suggested"]]}
    with open(os.path.join(hio.OUT_MODELS, "calcium_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    if os.path.exists(review_path()):                           # lesion ids changed: old review no longer applies
        os.remove(review_path())
    save_figures(hu, eat, lab3, lesions, os.path.join(hio.OUT_VIS, "calcium_overlay.png"))
    log(f"Done in {time.time() - t0:.1f}s -> automatic Agatston {auto} (review the {n_sug} lesion(s) in the app)")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=CAC_SERIES)
    main(ap.parse_args().series)
