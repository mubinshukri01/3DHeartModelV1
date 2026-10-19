"""
==============================================================
  STEP 1b - Epicardial fat (EAT) of the patient
  UTM-Claude Research Project | rohayanti@utm.my

  Runs eat_segment.py (trained and validated on the Rodriguez benchmark,
  see eat_benchmark.py) on the non-contrast CT CALCIUM SCORE series:
  EAT is normally measured on this kind of scan, and it is what the
  model was trained on. EAT volume and mean attenuation are measured
  there.

  For the 3D model the mask is moved onto the 1 mm CCTA grid of
  segment_heart.py. The two series share a frame of reference but were
  acquired in separate breath-holds, so a rigid registration (mutual
  information, around the heart) corrects the few-mm offset. The calcium
  scan is at 45 % R-R and the CCTA at 75 %, so in the 3D model the fat
  sits slightly off the heart surface in places. The measured volume
  does not depend on this mapping.

  Usage (after segment_heart.py):
    python eat_patient.py [--series "CT CALCIUM SCORE"]

  Output (output/3d_models):
    eat_labels.nii.gz   EAT on the CCTA 1 mm grid (same grid as heart_labels.nii.gz)
    eat_cac.nii.gz      EAT on the calcium-score grid (where it was measured)
    eat_stats.json      volume, attenuation, model and registration details
  Output (output/visualizations):
    eat_overlay.png
==============================================================
"""
import argparse, json, os, time

import numpy as np
import SimpleITK as sitk

import eat_segment as es
import heart_io as hio

EAT_SERIES = "CT CALCIUM SCORE"
REG_MARGIN_MM = 20       # registration looks at the CCTA heart plus this margin


def segment(img, model):
    """EAT mask in the image's own index order. The model expects slice 0 = most superior."""
    hu = np.maximum(sitk.GetArrayFromImage(img), -1024)      # outside-FOV padding (-3024) -> air
    flip = img.GetDirection()[8] > 0                         # k increases towards the head
    vol = hu[::-1].copy() if flip else hu
    out = es.segment_eat(vol, img.GetSpacing()[0], img.GetSpacing()[2], model=model)
    eat = out["eat"][::-1] if flip else out["eat"]
    return eat, hu


def register(moving, fixed, heart_labels):
    """Rigid transform mapping CCTA points to calcium-scan points (SimpleITK resampling convention)."""
    clamp = lambda im: sitk.Clamp(sitk.Cast(im, sitk.sitkFloat32), sitk.sitkFloat32, -1024, 1500)
    f, m = clamp(fixed), clamp(moving)
    mask = sitk.BinaryDilate(sitk.Cast(heart_labels > 0, sitk.sitkUInt8), [REG_MARGIN_MM] * 3)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2, seed=0)
    R.SetMetricFixedMask(mask); R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200, relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1]); R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(sitk.CenteredTransformInitializer(f, m, sitk.Euler3DTransform(),
                                                            sitk.CenteredTransformInitializerFilter.GEOMETRY), inPlace=False)
    tx = R.Execute(f, m)
    return sitk.Euler3DTransform(sitk.CompositeTransform(tx).GetNthTransform(0)), R.GetMetricValue()


def save_overlay(cac, cac_hu, eat, ccta, eat_ccta, path):
    """Top: calcium-score slices at 15/40/65/90 % of the EAT volume. Bottom: CCTA at the same heights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fa = sitk.GetArrayFromImage(ccta)
    cum = np.cumsum(eat.sum((1, 2))) / max(eat.sum(), 1)
    ks = [int(np.searchsorted(cum, q)) for q in (0.15, 0.40, 0.65, 0.90)]
    zs = [cac.TransformIndexToPhysicalPoint((0, 0, k))[2] for k in ks]
    kc = [int(np.clip(round((z - ccta.GetOrigin()[2]) / ccta.GetSpacing()[2]), 0, fa.shape[0] - 1)) for z in zs]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.4), facecolor="#111")
    for row, (vol, mask, lo, hi, title, idx) in enumerate([(cac_hu, eat, -160, 240, "Calcium score (measured)", ks),
                                                          (fa, eat_ccta, -160, 540, "CCTA (3D model)", kc)]):
        for ax, k, z in zip(axs[row], idx, zs):
            img = np.stack([np.clip((vol[k] - lo) / (hi - lo), 0, 1)] * 3, -1)
            img[mask[k]] = (1.0, 0.8, 0.1)
            ax.imshow(img); ax.axis("off")
            ax.set_title(f"{title}, z = {z:.0f} mm", color="w", fontsize=9)
    fig.text(0.5, 0.01, "yellow = epicardial fat (EAT)", color="w", ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path, dpi=70, facecolor=fig.get_facecolor()); plt.close(fig)


def main(series=EAT_SERIES, log=print):
    t0 = time.time()
    ccta_path = os.path.join(hio.OUT_MODELS, "heart_ct_1mm.nii.gz")
    labels_path = os.path.join(hio.OUT_MODELS, "heart_labels.nii.gz")
    if not (os.path.exists(ccta_path) and os.path.exists(labels_path)):
        raise FileNotFoundError("run segment_heart.py first (needs heart_ct_1mm.nii.gz and heart_labels.nii.gz)")
    folder = os.path.join(hio.DATASET, series)
    log(f"[1/4] Loading {series} and the EAT model ...")
    cac = hio.load_series(folder)
    model = es.load_model()
    log("[2/4] Segmenting epicardial fat ...")
    eat, hu = segment(cac, model)
    sp = cac.GetSpacing(); vox_ml = sp[0] * sp[1] * sp[2] / 1000.0
    volume, attenuation = float(eat.sum() * vox_ml), float(hu[eat].mean()) if eat.any() else float("nan")
    log(f"      EAT {volume:.1f} mL, mean {attenuation:.1f} HU")
    eat_img = sitk.GetImageFromArray(eat.astype(np.uint8)); eat_img.CopyInformation(cac)
    sitk.WriteImage(eat_img, os.path.join(hio.OUT_MODELS, "eat_cac.nii.gz"))
    log("[3/4] Registering calcium score -> CCTA (rigid) ...")
    ccta, labels = sitk.ReadImage(ccta_path, sitk.sitkFloat32), sitk.ReadImage(labels_path)
    tx, metric = register(cac, ccta, labels)
    moved = sitk.Resample(sitk.Cast(eat_img, sitk.sitkFloat32), labels, tx, sitk.sitkLinear, 0.0) > 0.5
    sitk.WriteImage(sitk.Cast(moved, sitk.sitkUInt8), os.path.join(hio.OUT_MODELS, "eat_labels.nii.gz"))
    ang = np.degrees(tx.GetParameters()[:3]); shift = tx.GetParameters()[3:]
    log(f"      shift ({shift[0]:+.1f}, {shift[1]:+.1f}, {shift[2]:+.1f}) mm, rotation ({ang[0]:+.1f}, {ang[1]:+.1f}, {ang[2]:+.1f})°")
    log("[4/4] Saving stats and overlay ...")
    ds = model.get("lopo_summary", {})
    import pydicom
    first = pydicom.dcmread(os.path.join(folder, sorted(os.listdir(folder))[0]), stop_before_pixels=True)
    stats = {"series": series, "cardiac_phase_percent": float(getattr(first, "NominalPercentageOfCardiacPhase", float("nan"))),
             "eat_volume_ml": round(volume, 1), "eat_mean_hu": round(attenuation, 1),
             "fat_range_hu": list(es.FAT_RANGE), "threshold": model.get("threshold", es.THRESHOLD),
             "voxel_mm": [round(v, 4) for v in sp],
             "model": {"trained_on_patients": len(model.get("trained_on", [])), "benchmark": "Rodriguez (leave-one-patient-out)",
                       "benchmark_dice_mean": round(ds.get("dice_mean", float("nan")), 3),
                       "benchmark_volume_loa_ml": [round(v, 1) for v in ds.get("volume_loa_ml", [])]},
             "registration_to_ccta": {"shift_mm": [round(v, 2) for v in shift], "rotation_deg": [round(v, 2) for v in ang],
                                      "mattes_mi": round(metric, 3)}}
    with open(os.path.join(hio.OUT_MODELS, "eat_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    save_overlay(cac, hu, eat, ccta, sitk.GetArrayFromImage(moved) > 0, os.path.join(hio.OUT_VIS, "eat_overlay.png"))
    log(f"Done in {time.time() - t0:.1f}s -> EAT {volume:.1f} mL, mean {attenuation:.1f} HU")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=EAT_SERIES)
    a = ap.parse_args()
    main(a.series)
