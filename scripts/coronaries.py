"""
==============================================================
  STEP 1e (optional) - Coronary arteries with TotalSegmentator
  UTM-Claude Research Project | rohayanti@utm.my

  TotalSegmentator's 'coronary_arteries' task (licensed; a free licence for
  non-commercial use, set with totalseg_set_license) on the CCTA at its
  native resolution. The arteries are only 2-4 mm wide, so their mask stays
  on the native grid (cropped to the heart; TotalSegmentator predicts them
  at 0.7 mm) for their mesh;
  build_heart_model.py also puts them on the 1 mm grid as a label (slice
  tool, pointer labels, motion).

  Usage (after segment_heart.py):
    python coronaries.py [--series "CT CCTA DL"] [--totalseg PATH] [--device gpu|cpu]
  Output (output/3d_models):
    coronaries_dl.nii.gz   the arteries on the native CT grid, cropped to the heart
    coronaries_dl.json     volume, pieces, versions, run time
==============================================================
"""
import argparse, json, os, tempfile, time

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

import heart_io as hio
import segment_heart_dl as dl

TASK = "coronary_arteries"
CROP_MARGIN_MM = 10.0        # around the heart (heart_labels.nii.gz)
MIN_PIECE_MM3 = 2.0          # drop specks smaller than this


def coronaries_file():
    return os.path.join(hio.OUT_MODELS, "coronaries_dl.nii.gz")


def stats_file():
    return os.path.join(hio.OUT_MODELS, "coronaries_dl.json")


def crop_to(img, ref, margin_mm=CROP_MARGIN_MM):
    """The part of img (any grid) inside ref's labelled bounding box widened by margin_mm."""
    arr = sitk.GetArrayViewFromImage(ref)
    nz = np.argwhere(arr > 0)
    corners = [ref.TransformIndexToPhysicalPoint([int(v) for v in idx[::-1]]) for idx in (nz.min(0), nz.max(0))]
    lo = np.minimum(*map(np.array, corners)) - margin_mm
    hi = np.maximum(*map(np.array, corners)) + margin_mm
    a = np.array(img.TransformPhysicalPointToContinuousIndex(lo.tolist()))
    b = np.array(img.TransformPhysicalPointToContinuousIndex(hi.tolist()))
    start = np.clip(np.floor(np.minimum(a, b)).astype(int), 0, np.array(img.GetSize()) - 1)
    stop = np.clip(np.ceil(np.maximum(a, b)).astype(int) + 1, 1, np.array(img.GetSize()))
    return sitk.RegionOfInterest(img, (stop - start).tolist(), start.tolist())


def clean(mask, spacing_zyx):
    """Drop pieces smaller than MIN_PIECE_MM3."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask, 0
    sizes = ndi.sum(mask, lab, range(1, n + 1)) * float(np.prod(spacing_zyx))
    keep = np.flatnonzero(sizes >= MIN_PIECE_MM3) + 1
    return np.isin(lab, keep), len(keep)


def load(log=print):
    """The arteries' mask image (native grid, cropped), or None if coronaries.py has not been run."""
    return sitk.ReadImage(coronaries_file()) if os.path.exists(coronaries_file()) else None


def main(series=hio.DEFAULT_SERIES, totalseg=None, device="gpu", log=print):
    t0 = time.time()
    ref_path = os.path.join(hio.OUT_MODELS, "heart_labels.nii.gz")
    if not os.path.exists(ref_path):
        raise FileNotFoundError("run segment_heart.py first (needs heart_labels.nii.gz)")
    ts = dl.find_totalseg(totalseg)
    version = dl.ts_version(ts)
    if not dl.has_licence(ts):
        raise RuntimeError(f"'{TASK}' needs a TotalSegmentator licence (free for non-commercial use): "
                           "https://backend.totalsegmentator.com/license-academic/, then totalseg_set_license -l <key>")
    log(f"[1/3] TotalSegmentator {version}: exporting {series} at native resolution ...")
    with tempfile.TemporaryDirectory(prefix="heart_coronaries_") as tmp:     # native CT + raw output, deleted after
        ct = sitk.Clamp(hio.load_series(os.path.join(hio.DATASET, series)), sitk.sitkFloat32, -1024.0, 32767.0)
        ct_path, raw = os.path.join(tmp, "ct.nii.gz"), os.path.join(tmp, "coronaries.nii.gz")
        sitk.WriteImage(ct, ct_path)
        log(f"      {ct.GetSize()} @ {tuple(round(s, 3) for s in ct.GetSpacing())} mm")
        log(f"[2/3] Running the '{TASK}' task ...")
        t = time.time()
        dl.run_task(ts, ct_path, raw, TASK, device)
        log(f"      {time.time() - t:.0f}s")
        out = sitk.ReadImage(raw)
    log("[3/3] Cropping to the heart and saving ...")
    out = crop_to(out, sitk.ReadImage(ref_path))
    sp = out.GetSpacing()[::-1]
    mask, pieces = clean(sitk.GetArrayFromImage(out) > 0, sp)
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.CopyInformation(out)
    sitk.WriteImage(img, coronaries_file())
    stats = {"series": series, "totalsegmentator": version, "task": TASK, "device": device,
             "grid": {"size": list(img.GetSize()), "spacing_mm": [round(s, 4) for s in img.GetSpacing()]},
             "volume_ml": round(float(mask.sum() * np.prod(sp)) / 1000.0, 2), "pieces": pieces,
             "runtime_s": round(time.time() - t0)}
    with open(stats_file(), "w") as fh:
        json.dump(stats, fh, indent=2)
    log(f"Done in {stats['runtime_s']}s -> {coronaries_file()}: {stats['volume_ml']} mL in {pieces} piece(s)")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=hio.DEFAULT_SERIES)
    ap.add_argument("--totalseg", help="TotalSegmentator executable (default: HEART3D_TOTALSEG, ~/venvs/heart3d-dl, PATH)")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    a = ap.parse_args()
    main(a.series, a.totalseg, a.device)
