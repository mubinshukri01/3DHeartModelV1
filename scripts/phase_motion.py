"""
==============================================================
  STEP 11 - Measured motion from a second cardiac phase
  UTM-Claude Research Project | rohayanti@utm.my

  The 3D model's motion is procedural (build_heart_model.py), because
  the dataset has one contrast phase: CCTA at 75 % R-R. The only other
  phase is the non-contrast CT CALCIUM SCORE at 45 % R-R (close to end-
  systole), from a separate breath-hold. This script measures how the
  heart differs between the two and compares that with the procedural
  model at 45 %:
    1  rigid registration (eat_patient.register) removes the breath-hold
       offset
    2  B-spline registration (Mattes mutual information, heart + margin)
       gives the deformation from 75 % to 45 %
    3  on the heart's outer surface, the measured displacement is compared
       with the procedural morph targets at 45 % (weights W(45) - W(75)),
       and the whole-heart volume change with that of the model
  A breath-hold shift cannot be told apart from rigid motion of the heart,
  so the best-fitting rigid motion is removed from both sides before they
  are compared.

  Without contrast, blood and myocardium look alike at 45 %, so only the
  outer surface (against fat and lung) can be measured; the chamber walls
  inside cannot. --validate tests this on synthetic motion with a known
  answer: the procedural field applied to the CCTA, with and without
  simulated contrast.

  Needs segment_heart.py first (heart_ct_1mm.nii.gz, heart_labels.nii.gz).
  Usage:
    python phase_motion.py [--series "CT CALCIUM SCORE"] [--validate]
  Output (output/3d_models):      phase_motion.json (+ phase_motion_validation.json)
                                  phase_motion_bspline.tfm
  Output (output/visualizations): phase_motion.png
==============================================================
"""
import argparse, json, os, time

import numpy as np
import pydicom
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi

import build_heart_model as bhm
import eat_patient
import heart_io as hio

PHASE_SERIES = "CT CALCIUM SCORE"
REST_PHASE = 0.75           # CCTA phase (the model's rest shape)
CROP_MM = 30                # registration domain: heart bounding box + this
MARGIN_MM = 15              # metric mask: heart + this, so the outer surface (fat, lung) is included
GRID_MM = 20.0              # B-spline control-point spacing
SHRINK = [4, 2, 1]          # image pyramid down to 1 mm (stopping at 2 mm overestimated a phantom's motion by 40 %)
SAMPLING = [0.2, 0.1, 0.05]  # metric samples per level: same phantom accuracy as 20 % throughout, ~5x faster
CLAMP_HU = (-1024.0, 400.0)  # intensity range seen by the metric
SURFACE_FACES = 60000       # face budget of the outer-surface mesh
FALLOFF_MM = 10.0           # validation: the synthetic motion fades out this fast outside the heart
NONCONTRAST_HU = 45.0       # validation: blood and myocardium without contrast
OUT_JSON = os.path.join(hio.OUT_MODELS, "phase_motion.json")
OUT_VAL = os.path.join(hio.OUT_MODELS, "phase_motion_validation.json")
OUT_TFM = os.path.join(hio.OUT_MODELS, "phase_motion_bspline.tfm")
OUT_FIG = os.path.join(hio.OUT_VIS, "phase_motion.png")


# ── geometry helpers ──────────────────────────────────────────────────────
def gltf_to_lps(p):
    """glTF (m) -> patient LPS (mm); inverse of build_heart_model.lps_to_gltf (also for displacements)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1) * 1000.0


def phase_weights(phase):
    """Morph weights of build_heart_model.cycle_weights at an R-R phase (0..1)."""
    ph, W = bhm.cycle_weights(1000)
    return np.array([np.interp(phase, ph, W[:, i]) for i in range(W.shape[1])])


def remove_rigid(p, d):
    """Displacements d at points p minus their best-fitting rigid motion (least squares, Kabsch)."""
    q = p + d
    pc, qc = p.mean(0), q.mean(0)
    U, _, Vt = np.linalg.svd((p - pc).T @ (q - qc))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return q - ((p - pc) @ R.T + qc)


def sample_field(field, pts_lps):
    """Displacement field image (LPS mm) sampled at LPS points (linear)."""
    arr = sitk.GetArrayViewFromImage(field)                      # (Z, Y, X, 3)
    idx = ((pts_lps - np.array(field.GetOrigin())) / np.array(field.GetSpacing()))[:, ::-1].T
    return np.stack([ndi.map_coordinates(arr[..., c], idx, order=1, mode="nearest") for c in range(3)], 1)


def field_of(transform, ref):
    return sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat64, ref.GetSize(), ref.GetOrigin(),
                                             ref.GetSpacing(), ref.GetDirection())


# ── registration ─────────────────────────────────────────────────────────
def crop_to_heart(img, heart, margin_mm=CROP_MM):
    """Region of img around the heart mask (same grid), plus the mask cropped the same way."""
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(heart)
    x0, y0, z0, sx, sy, sz = ls.GetBoundingBox(1)
    m = [int(round(margin_mm / s)) for s in img.GetSpacing()]
    lo = [max(a - b, 0) for a, b in zip((x0, y0, z0), m)]
    hi = [min(a + s + b, n) for a, s, b, n in zip((x0, y0, z0), (sx, sy, sz), m, img.GetSize())]
    size = [h - l for l, h in zip(lo, hi)]
    return sitk.RegionOfInterest(img, size, lo), sitk.RegionOfInterest(heart, size, lo)


def deformable(fixed, moving, heart, rigid, grid_mm=GRID_MM, log=print):
    """B-spline transform on top of `rigid`: maps fixed (75 %) points, in the rigidly aligned frame, to their
    position at the moving phase. fixed/heart are already cropped; heart is a UInt8 mask on that grid."""
    clamp = lambda im: sitk.Clamp(sitk.Cast(im, sitk.sitkFloat32), sitk.sitkFloat32, *CLAMP_HU)
    f, m = clamp(fixed), clamp(moving)
    mask = sitk.BinaryDilate(heart, [int(round(MARGIN_MM / s)) for s in fixed.GetSpacing()])
    mesh = [max(1, int(round(n * s / grid_mm))) for n, s in zip(fixed.GetSize(), fixed.GetSpacing())]
    tx = sitk.BSplineTransformInitializer(f, mesh, 3)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentagePerLevel(SAMPLING, seed=0)
    R.SetMetricFixedMask(mask)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100,
                           maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000,
                           costFunctionConvergenceFactor=1e7)
    R.SetMovingInitialTransform(rigid)
    R.SetInitialTransform(tx, inPlace=True)       # one grid: L-BFGS-B cannot follow a grid refined per level
    R.SetShrinkFactorsPerLevel(SHRINK)
    R.SetSmoothingSigmasPerLevel([s / 2 for s in SHRINK])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    out = R.Execute(f, m)
    log(f"      B-spline mesh {mesh} ({grid_mm:.0f} mm), Mattes MI {R.GetMetricValue():.3f}, "
        f"{R.GetOptimizerIteration()} iterations (last level)")
    return sitk.BSplineTransform(out), R.GetMetricValue()


# ── surfaces and the procedural model ────────────────────────────────────
def outer_surface(lab_img, labels, faces=SURFACE_FACES):
    """Whole-heart outer surface (glTF m), closed."""
    return bhm.mask_to_mesh(labels > 0, lab_img, faces)


def procedural(ctx, pts, dw, generic=False):
    """Procedural displacement (glTF m) at points: the model's blended morph targets, or (generic) the
    ventricle/atrium field of motion_targets' default branch everywhere (the validation's known motion)."""
    T = (bhm.motion_targets(ctx["motion"], 0, pts, None, None, ctx["aorta_center"], None, None) if generic
         else ctx["targets"](pts))
    return sum(w * t for w, t in zip(dw, T))


def procedural_volume_ratio(ctx, surf, dw):
    """Volume inside the heart's outer surface at the phase / at rest, under the model's motion (the
    blended field is continuous, so this is the model's mean Jacobian over the heart, like the measurement)."""
    return abs(trimesh.Trimesh(surf.vertices + procedural(ctx, surf.vertices, dw), surf.faces,
                               process=False).volume) / abs(surf.volume)


def compare(pts_gltf, meas_lps, proc_gltf, ctx):
    """Non-rigid surface motion, measured vs procedural (mm), overall and by level along the LV axis."""
    p = gltf_to_lps(pts_gltf)
    dm = remove_rigid(p, meas_lps)
    dp = remove_rigid(p, gltf_to_lps(proc_gltf))
    u = gltf_to_lps(ctx["motion"].u) / 1000.0                     # unit LV axis (apex -> base), LPS
    apex = gltf_to_lps(ctx["motion"].A)
    s = ((p - apex) @ u) / (ctx["motion"].L * 1000.0)             # 0 apex .. 1 mitral plane
    nm, npr = np.linalg.norm(dm, axis=1), np.linalg.norm(dp, axis=1)
    levels ={"apex (s<0.33)": s < 0.33, "mid (0.33-0.66)": (s >= 0.33) & (s < 0.66),
              "base (0.66-1)": (s >= 0.66) & (s <= 1.0), "atria / vessels (s>1)": s > 1.0}
    by_level = {name: {"vertices": int(sel.sum()),
                       "measured_mm": round(float(nm[sel].mean()), 2), "procedural_mm": round(float(npr[sel].mean()), 2),
                       "measured_long_mm": round(float((dm[sel] @ u).mean()), 2),
                       "procedural_long_mm": round(float((dp[sel] @ u).mean()), 2)}
                for name, sel in levels.items() if sel.any()}
    return {"mean_measured_mm": round(float(nm.mean()), 2), "mean_procedural_mm": round(float(npr.mean()), 2),
            "mean_difference_mm": round(float(np.linalg.norm(dm - dp, axis=1).mean()), 2),
            "direction_agreement_cos": round(agreement(dm, dp), 3),
            "by_level": by_level}, dm, dp, s


# ── validation on synthetic motion ───────────────────────────────────────
def synthetic_field(ctx, ref, heart_arr, dw, step=2):
    """Known smooth displacement (LPS mm) on the ref grid: the model's generic morph targets at the
    phase, fading out FALLOFF_MM outside the heart. Evaluated every `step` voxels, then interpolated."""
    Z, Y, X = heart_arr.shape
    zz, yy, xx = np.meshgrid(*(np.arange(0, n, step) for n in (Z, Y, X)), indexing="ij")
    idx = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1).astype(float)
    lps = idx * np.array(ref.GetSpacing()) + np.array(ref.GetOrigin())
    d = gltf_to_lps(procedural(ctx, bhm.lps_to_gltf(lps), dw, generic=True))
    out = ndi.distance_transform_edt(~heart_arr, sampling=ref.GetSpacing()[::-1])
    fade = np.exp(-out[::step, ::step, ::step].ravel() / FALLOFF_MM)
    coarse = (d * fade[:, None]).reshape(zz.shape + (3,))
    full = np.stack([ndi.zoom(coarse[..., c], [n / m for n, m in zip((Z, Y, X), zz.shape)], order=1)
                     for c in range(3)], -1)[:Z, :Y, :X]
    img = sitk.GetImageFromArray(full.astype(np.float64), isVector=True)
    img.CopyInformation(ref)
    return img


def warp_by(img, truth, default=-1024.0):
    """Image of the other phase: a 75 % point x moves to x + truth(x), so moving(x + truth(x)) = img(x)."""
    inv = sitk.InvertDisplacementField(truth, maximumNumberOfIterations=50, maxErrorToleranceThreshold=0.01,
                                       meanErrorToleranceThreshold=0.001, enforceBoundaryCondition=True)
    return sitk.Resample(img, img, sitk.DisplacementFieldTransform(sitk.Cast(inv, sitk.sitkVectorFloat64)),
                         sitk.sitkLinear, default)


def remove_contrast(img, heart, seed=0):
    """Simulated non-contrast scan: inside the (dilated) heart, blood and myocardium become ~45 HU."""
    a = sitk.GetArrayFromImage(img)
    zone = ndi.binary_dilation(sitk.GetArrayFromImage(heart) > 0, iterations=2) & (a > 0)
    a[zone] = NONCONTRAST_HU + np.random.default_rng(seed).normal(0, 15, zone.sum())
    out = sitk.GetImageFromArray(a)
    out.CopyInformation(img)
    return out


def agreement(a, b):
    """Direction agreement of two displacement sets: cosine per point, weighted by magnitude."""
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return float(np.average(np.sum(a * b, 1) / np.maximum(na * nb, 1e-9), weights=na + nb))


def volume_ratio(field, heart):
    """Mean Jacobian determinant of a displacement field over the heart = volume at the phase / at rest."""
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field))
    return float(jac[sitk.GetArrayFromImage(heart) > 0].mean())


def validate(ctx, fixed, heart, lab_img, labels, dw, log=print):
    """Known answers for the measurement: the model's motion applied to the CCTA (with contrast, and with the
    contrast removed like the real 45 % scan), and no motion at all with contrast removed (the noise floor)."""
    truth = synthetic_field(ctx, fixed, sitk.GetArrayFromImage(heart) > 0, dw)
    zero = sitk.GetImageFromArray(np.zeros(sitk.GetArrayViewFromImage(truth).shape), isVector=True)
    zero.CopyInformation(fixed)
    moved = warp_by(fixed, truth)
    moved_heart = sitk.Resample(heart, heart, sitk.DisplacementFieldTransform(sitk.Cast(
        sitk.InvertDisplacementField(truth), sitk.sitkVectorFloat64)), sitk.sitkNearestNeighbor, 0)
    surf = outer_surface(lab_img, labels)
    lv = bhm.mask_to_mesh(labels == 1, lab_img, 20000)
    cases = [("model motion, with contrast", moved, truth),
             ("model motion, contrast removed", remove_contrast(moved, moved_heart), truth),
             ("no motion, contrast removed", remove_contrast(fixed, heart, seed=1), zero)]
    res = {}
    for name, moving, true_field in cases:
        t = time.time()
        rigid = eat_patient.register(moving, fixed, heart)[0]
        tx, metric = deformable(fixed, moving, heart, rigid, log=log)
        full = field_of(sitk.CompositeTransform([rigid, tx]), fixed)
        row = {}
        for part, pts in [("outer surface", surf.vertices), ("LV cavity wall (inside)", lv.vertices)]:
            p = gltf_to_lps(pts)
            est, true = remove_rigid(p, sample_field(full, p)), remove_rigid(p, sample_field(true_field, p))
            err = np.linalg.norm(est - true, axis=1)
            row[part] = {"true_mm": round(float(np.linalg.norm(true, axis=1).mean()), 2),
                         "measured_mm": round(float(np.linalg.norm(est, axis=1).mean()), 2),
                         "error_mean_mm": round(float(err.mean()), 2),
                         "error_p95_mm": round(float(np.percentile(err, 95)), 2),
                         "direction_agreement_cos": round(agreement(est, true), 3) if np.abs(true).max() > 0 else None}
        row["whole_heart_volume_ratio"] = {"true": round(volume_ratio(true_field, heart), 3),
                                           "measured": round(volume_ratio(full, heart), 3)}
        res[name] = row
        log(f"      {name} ({time.time() - t:.0f}s): {row}")
    return res


# ── figure ────────────────────────────────────────────────────────────────
def save_figure(fixed, moving_aligned, heart, heart_45, pts, dm, dp, s, u, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = sitk.GetArrayFromImage(moving_aligned)
    h75, h45 = sitk.GetArrayFromImage(heart) > 0, sitk.GetArrayFromImage(heart_45) > 0
    zs = np.where(h75.any((1, 2)))[0]
    levels = np.linspace(zs.min(), zs.max(), 5)[1:-1].astype(int)
    yc = int(np.mean(np.where(h75)[1]))
    edge = lambda m: m & ~ndi.binary_erosion(m, iterations=1)
    fig, axs = plt.subplots(2, 4, figsize=(18, 9.5), facecolor="#111")
    views = [(f"axial z={z}", img[z], h75[z], h45[z]) for z in levels]
    views.append(("coronal", img[:, yc][::-1], h75[:, yc][::-1], h45[:, yc][::-1]))
    for ax, (t, im, a, b) in zip(axs[0], views):
        rgb = np.stack([np.clip((im + 200) / 500, 0, 1)] * 3, -1)
        rgb[edge(a)] = (0.2, 1.0, 0.3)
        rgb[edge(b)] = (1.0, 0.25, 0.25)
        ax.imshow(rgb); ax.axis("off")
        ax.set_title(f"45 % scan (rigidly aligned), {t}\ngreen = heart at 75 %, red = registered to 45 %",
                     color="w", fontsize=8)
    order = np.argsort(pts[:, 2])    # anterior view (glTF +Z = anterior): back first, front drawn last
    x, y = pts[order, 0], pts[order, 1]   # patient left to the right, superior up
    lim = np.percentile(np.linalg.norm(np.vstack([dm, dp]), axis=1), 99)
    for ax, d, t in [(axs[1, 0], dm, "measured"), (axs[1, 1], dp, "procedural model")]:
        sc = ax.scatter(x, y, c=np.linalg.norm(d[order], axis=1), s=1, cmap="magma", vmin=0, vmax=lim)
        ax.set_aspect("equal"); ax.set_facecolor("#111"); ax.tick_params(colors="w")
        ax.set_title(f"Outer surface, 75 % -> 45 %, {t}\n(non-rigid part, mm; anterior view)", color="w", fontsize=9)
    fig.colorbar(sc, ax=axs[1, :2], fraction=0.03).ax.tick_params(colors="w")
    bins = np.linspace(-0.1, 1.5, 17)
    mid = (bins[:-1] + bins[1:]) / 2
    for ax, comp, t in [(axs[1, 2], lambda d: d @ u, "longitudinal (+ towards base)"),
                        (axs[1, 3], lambda d: np.linalg.norm(d, axis=1), "magnitude")]:
        for d, c, n in [(dm, "#e74c3c", "measured"), (dp, "#3498db", "procedural")]:
            v = comp(d)
            mean = [v[(s >= a) & (s < b)].mean() if ((s >= a) & (s < b)).sum() > 20 else np.nan
                    for a, b in zip(bins[:-1], bins[1:])]
            ax.plot(mid, mean, color=c, lw=2, label=n)
        ax.axvline(1.0, color="#999", ls=":")
        ax.set_facecolor("#111"); ax.tick_params(colors="w"); ax.legend(fontsize=8)
        ax.set_xlabel("height along LV axis (0 apex, 1 mitral plane)", color="w")
        ax.set_title(f"Surface motion by level: {t} (mm)", color="w", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=70, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────
def main(series=PHASE_SERIES, run_validation=False, log=print):
    t0 = time.time()
    ref_path = os.path.join(hio.OUT_MODELS, "heart_ct_1mm.nii.gz")
    lab_path = os.path.join(hio.OUT_MODELS, "heart_labels.nii.gz")
    if not (os.path.exists(ref_path) and os.path.exists(lab_path)):
        raise FileNotFoundError("run segment_heart.py first (needs heart_ct_1mm.nii.gz and heart_labels.nii.gz)")
    folder = os.path.join(hio.DATASET, series)
    first = pydicom.dcmread(os.path.join(folder, sorted(os.listdir(folder))[0]), stop_before_pixels=True)
    phase = float(first.NominalPercentageOfCardiacPhase) / 100.0
    ccta, lab_img = sitk.ReadImage(ref_path, sitk.sitkFloat32), sitk.ReadImage(lab_path)
    labels = sitk.GetArrayFromImage(lab_img)       # the heart outline: the rule labels, as measured in step 11
    with open(os.path.join(hio.OUT_MODELS, "segmentation_stats.json")) as fh:   # the motion: the model's
        model = bhm.model_labels(lab_img, log=lambda *_: None)[0]
        axis = bhm.lv_long_axis(model, lab_img) if bhm.chamber_source(lab_img) == "dl" else json.load(fh)["lv_long_axis_lps_mm"]
        ctx = bhm.motion_context(lab_img, model, axis)
    dw = phase_weights(phase) - phase_weights(REST_PHASE)
    log(f"[1/4] {series} at {phase:.0%} R-R; procedural weights W({phase:.0%}) - W({REST_PHASE:.0%}) = "
        f"{np.round(dw, 3).tolist()} ({', '.join(bhm.TARGETS)})")
    heart_full = sitk.Cast(lab_img > 0, sitk.sitkUInt8)
    fixed, heart = crop_to_heart(ccta, heart_full)
    lab_crop = sitk.RegionOfInterest(lab_img, fixed.GetSize(), lab_img.TransformPhysicalPointToIndex(fixed.GetOrigin()))
    val = None
    if run_validation:
        log("[v] Validation on synthetic motion (known answer) ...")
        val = validate(ctx, fixed, heart, lab_crop, sitk.GetArrayFromImage(lab_crop), dw, log)
        with open(OUT_VAL, "w") as fh:
            json.dump({"phase": phase, "weights_delta": dw.tolist(), "results": val}, fh, indent=2)
    log(f"[2/4] Registering {phase:.0%} -> {REST_PHASE:.0%} (rigid, then B-spline) ...")
    t = time.time()
    moving = sitk.Clamp(hio.load_series(folder), sitk.sitkFloat32, -1024.0, 1e6)   # outside-FOV padding -> air
    rigid, rigid_mi = eat_patient.register(moving, ccta, lab_img)
    log(f"      rigid: {time.time() - t:.0f}s")
    t = time.time()
    tx, metric = deformable(fixed, moving, heart, rigid, log=log)
    sitk.WriteTransform(tx, OUT_TFM)
    field = field_of(tx, fixed)                                   # non-rigid part only (aligned frame)
    vol_ratio = volume_ratio(field, heart)
    log(f"      B-spline: {time.time() - t:.0f}s")
    log("[3/4] Comparing the outer surface with the procedural model ...")
    t = time.time()
    surf = outer_surface(lab_img, labels)
    meas = sample_field(field, gltf_to_lps(surf.vertices))
    proc = procedural(ctx, surf.vertices, dw)
    cmp, dm, dp, s = compare(surf.vertices, meas, proc, ctx)
    proc_ratio = procedural_volume_ratio(ctx, surf, dw)
    log(f"      {len(surf.vertices)} surface vertices: {time.time() - t:.0f}s")
    t = time.time()
    ang = np.degrees(rigid.GetParameters()[:3]); shift = rigid.GetParameters()[3:]
    stats = {"series": series, "phase_percent": round(phase * 100), "rest_phase_percent": round(REST_PHASE * 100),
             "weights_delta": dict(zip(bhm.TARGETS, np.round(dw, 4).tolist())),
             "rigid": {"shift_mm": [round(v, 2) for v in shift], "rotation_deg": [round(v, 2) for v in ang],
                       "mattes_mi": round(rigid_mi, 3)},
             "bspline": {"grid_mm": GRID_MM, "mattes_mi": round(metric, 3)},
             "whole_heart_volume_ratio": {"measured": round(vol_ratio, 3), "procedural": round(proc_ratio, 3)},
             "outer_surface": cmp,
             "validation": val,
             "runtime_s": round(time.time() - t0)}
    with open(OUT_JSON, "w") as fh:
        json.dump(stats, fh, indent=2)
    log("[4/4] Saving figure ...")
    aligned = sitk.Resample(moving, fixed, rigid, sitk.sitkLinear, -1024.0)
    inv = sitk.InvertDisplacementField(field)
    heart_45 = sitk.Resample(heart, heart, sitk.DisplacementFieldTransform(sitk.Cast(inv, sitk.sitkVectorFloat64)),
                             sitk.sitkNearestNeighbor, 0)
    save_figure(fixed, aligned, heart, heart_45, surf.vertices, dm, dp, s,
                gltf_to_lps(ctx["motion"].u) / 1000.0, OUT_FIG)
    log(f"      figure: {time.time() - t:.0f}s")
    log(f"Done in {stats['runtime_s']}s -> whole heart V45/V75 measured {vol_ratio:.3f} vs model {proc_ratio:.3f}; "
        f"surface motion {cmp['mean_measured_mm']} vs {cmp['mean_procedural_mm']} mm, "
        f"direction agreement {cmp['direction_agreement_cos']}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=PHASE_SERIES)
    ap.add_argument("--validate", action="store_true", help="also run three known-answer checks (~10 min more)")
    a = ap.parse_args()
    main(a.series, a.validate)
