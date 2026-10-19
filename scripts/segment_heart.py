"""
==============================================================
  STEP 1 - Segment the 3D heart from patient CCTA (DICOM)
  UTM-Claude Research Project | rohayanti@utm.my

  Classical (training-free) HU-based segmentation tuned for a
  coronary-timed contrast CT (left heart bright, right heart dim):

    1  Whole heart  = contrast-enhanced soft tissue (>110 HU after
                      smoothing) connected to the LV, with liver,
                      spine and descending aorta cut away
    2  Left blood pool (>300 HU) split by watershed into
                      LV cavity / LA / Aorta
    3  Remainder    = Myocardium + right heart
    4  Remainder split by anatomy (split_right_heart): LV myocardium
                      (shell around the LV) + thin LA/aortic walls,
                      right atrium, right ventricle, pulmonary artery,
                      superior vena cava. Approximate: the saline flush
                      leaves right-heart blood as dim as the muscle, so
                      this uses position and valve narrowings, not HU.

  Usage:
    python segment_heart.py                     # CT CCTA DL series
    python segment_heart.py --series "CT SSFreeze 116 Original Series 302"

  Output (output/3d_models):
    heart_labels.nii.gz, heart_ct_1mm.nii.gz, segmentation_stats.json
  Output (output/visualizations):
    segmentation_overlay.png
==============================================================
"""
import argparse, json, os, time
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.filters import sobel

import heart_io as hio

# ── Tunable parameters ────────────────────────────────────────────────────
# Calibrated on TC_PTID (GE Revolution, SmartPrep CCTA, 75% R-R). HU values
# are compared after smoothing; lengths are voxels (= mm at the default
# 1 mm spacing). After changing any of these, re-run and check
# output/visualizations/segmentation_overlay.png.
SMOOTH_SIGMA = 1.5        # Gaussian sigma applied after a 3-voxel median filter
CONTRAST_LOWER = 300      # HU  iodinated blood pool (LV, LA, aorta) ...
CONTRAST_UPPER = 700      # HU  ... anything brighter is bone or calcium
ENVELOPE_LOWER = 110      # HU  whole heart: contrast-enhanced myocardium + right heart ...
ENVELOPE_UPPER = 700      # HU  ... (fat, lung and unenhanced tissue fall below the lower bound)
BONE_FILL_HU = 600        # HU  cortical bone; regions it encloses (vertebral bodies) are filled
BONE_HU = 900             # HU  always treated as bone
EXCLUDE_MIN_VOX = 5000    # other bright blobs bigger than this (descending aorta, IVC) are cut away
EROSION_RADIUS = 4        # erosion that breaks thin bridges from the heart to liver / chest wall
LV_APEX_SLAB = 15         # slices above the lowest pool slice that seed the LV
AORTA_TOP_SLAB = 8        # slices below the top of the pool that seed the aorta
MITRAL_CUT_FACTOR = 1.1   # LV/LA plane sits this many LA radii before the LA centre
# right heart (step 4, lengths in mm)
LV_WALL_MM = 10           # LV myocardium = remainder this close to the LV pool (ventricular level)
THIN_WALL_MM = 3          # LA and aortic wall kept as myocardium
RH_MIN_ML = 2.0           # right-heart fragments smaller than this stay myocardium
SVC_ABOVE_LA_MM = 12      # SVC-RA junction this far above the LA centre
PA_TOP_MM = 12            # PA seeds: remainder left of the aorta in the top slices

LABELS = {  # id: (name, RGBA 0-1); left heart in reds, right heart in blues
    1: ("LV blood pool", (0.85, 0.10, 0.12, 1.0)),
    2: ("LA blood pool", (0.95, 0.45, 0.50, 1.0)),
    3: ("Aorta",         (0.98, 0.75, 0.20, 1.0)),
    4: ("Myocardium",    (0.62, 0.20, 0.28, 0.45)),
    5: ("Right atrium",       (0.45, 0.62, 0.95, 1.0)),
    6: ("Right ventricle",    (0.20, 0.35, 0.85, 1.0)),
    7: ("Pulmonary artery",   (0.50, 0.38, 0.85, 1.0)),
    8: ("Superior vena cava", (0.15, 0.25, 0.60, 1.0)),
}
SLUGS = {1: "lv", 2: "la", 3: "aorta", 4: "myocardium", 5: "ra", 6: "rv", 7: "pa", 8: "svc"}  # mesh file names


def ball(r):
    g = np.arange(-r, r + 1) ** 2
    return np.add.outer(np.add.outer(g, g), g) <= r * r


def largest_cc(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def segment(hu, log=print):
    """hu: (Z,Y,X) array at 1 mm isotropic, z = inferior->superior."""
    g = ndi.gaussian_filter(ndi.median_filter(hu, 3), SMOOTH_SIGMA)

    # --- bright (iodinated) blood and bone -----------------------------
    bright = ndi.binary_opening((g > CONTRAST_LOWER) & (g < CONTRAST_UPPER), ball(1))
    lab, n = ndi.label(bright)
    sizes = ndi.sum(bright, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    cy, cx = hu.shape[1] / 2, hu.shape[2] / 2
    # heart pool = large component closest to image centre (in-plane)
    best, best_score = None, -1
    for i in order[:6]:
        com = ndi.center_of_mass(lab == i + 1)
        d = np.hypot(com[1] - cy, com[2] - cx)
        score = sizes[i] / (1 + d / 20)
        if score > best_score:
            best, best_score = i + 1, score
    pool = lab == best
    # other big bright tubes (descending aorta, IVC...) to exclude
    others = np.isin(lab, [i + 1 for i in order[:6] if i + 1 != best and sizes[i] > EXCLUDE_MIN_VOX])
    bone = ndi.binary_fill_holes(hu > BONE_FILL_HU) | (hu > BONE_HU)
    bone = ndi.binary_closing(bone, ball(3))
    # vertebral body fill per axial slice
    bone = np.stack([ndi.binary_fill_holes(s) for s in bone])
    bone &= ~ndi.binary_dilation(pool, ball(3))   # calcium / noise inside the pool is not bone
    excl = ndi.binary_dilation(others | bone, ball(2))
    log(f"  bright pool {pool.sum()} vox, excluded {excl.sum()} vox")

    # --- whole heart ---------------------------------------------------
    cand = (g > ENVELOPE_LOWER) & (g < ENVELOPE_UPPER) & ~excl
    r = EROSION_RADIUS
    ero = ndi.binary_erosion(cand, ball(r))
    lab2, _ = ndi.label(ero)
    ids, cnt = np.unique(lab2[pool & ero], return_counts=True)
    keep = ids[(ids > 0)]
    keep = keep[np.argmax(cnt[ids > 0])]
    core = lab2 == keep
    heart = ndi.binary_dilation(core, ball(r + 1)) & (cand | pool)
    heart = ndi.binary_closing(heart, ball(2))
    heart = np.stack([ndi.binary_fill_holes(s) for s in heart])
    heart = largest_cc(heart)

    # --- split left blood pool (LV / LA / Aorta) -------------------------
    lpool = pool & heart
    lpool = largest_cc(ndi.binary_opening(lpool, ball(2)))
    dist = ndi.distance_transform_edt(lpool)
    zs = np.where(lpool.any((1, 2)))[0]
    markers = np.zeros(hu.shape, np.int32)
    # LV marker: apical slab of the pool (LV is the most inferior left chamber)
    slab = lpool.copy(); slab[zs.min() + LV_APEX_SLAB:] = False
    markers[largest_cc(slab)] = 1
    # LA marker: thickest part of the pool (largest inscribed sphere, posterior)
    t = max(dist.max() - 4, 5)
    core = largest_cc(dist > t)
    markers[core & (markers == 0)] = 2
    # Aorta marker: most anterior (min y, LPS) blob in the top slices
    top = lpool.copy(); top[: zs.max() - AORTA_TOP_SLAB] = False
    tl, tn = ndi.label(top)
    if tn:
        ant = min(range(1, tn + 1), key=lambda k: ndi.center_of_mass(tl == k)[1]
                  if (tl == k).sum() > 50 else 1e9)
        markers[(tl == ant) & (markers == 0)] = 3
    # (a) Aorta vs left heart: watershed on edges (aortic valve) + distance
    m2 = np.where(markers == 3, 3, np.where(markers > 0, 1, 0))
    ws = watershed(sobel(ndi.gaussian_filter(hu, 1.0)) / 50 - dist, m2, mask=lpool)
    # (b) LV vs LA: plane perpendicular to the long axis (apex -> LA core),
    #     tangent to the largest sphere inscribed in the LA (~ mitral plane)
    left = ws == 1
    apex = np.array(ndi.center_of_mass(markers == 1))
    c_la = np.array(ndi.center_of_mass(markers == 2))
    L = np.linalg.norm(c_la - apex)
    u = (c_la - apex) / L
    pts = np.argwhere(left)
    t = (pts - apex) @ u
    t_cut = L - MITRAL_CUT_FACTOR * float(dist.max())
    lv_m = np.zeros_like(left); lv_m[tuple(pts[t < t_cut].T)] = True
    # beyond the plane: LA core vs LV face compete (watershed on -distance) so
    # the LV outflow tract, which also crosses the plane, stays with the LV
    far = left & ~lv_m
    face = ndi.binary_dilation(lv_m, ball(1)) & far
    mk = np.zeros(hu.shape, np.int32); mk[face] = 1; mk[core & far] = 2
    ws2 = watershed(-ndi.gaussian_filter(dist, 1), mk, mask=far)
    la_m = largest_cc(ws2 == 2)
    lv_m = largest_cc(left & ~la_m)
    la_m = left & ~lv_m
    log(f"  long axis {L:.0f} mm, mitral cut at {t_cut:.0f} mm from apex")

    labels = np.zeros(hu.shape, np.uint8)
    labels[heart] = 4
    labels[la_m] = 2
    labels[lv_m] = 1
    labels[ws == 3] = 3
    split_right_heart(labels, apex, u, t_cut, c_la, log)
    # attach the long-axis direction for the animation step
    segment.long_axis = dict(apex=[float(v) for v in apex], base=[float(v) for v in apex + u * t_cut],
                             la_center=[float(v) for v in c_la])
    return labels


def split_right_heart(labels, apex, u, L, la_center, log=print):
    """Split label 4 (myocardium + right heart) in place into myocardium (4), RA (5), RV (6), PA (7), SVC (8).

    Index space (z, y, x) at 1 mm; LPS: +x patient left, +y posterior, z inferior -> superior.
    apex, u, L: LV long axis (unit vector, apex -> mitral plane distance); s = 0 apex .. 1 mitral plane.
    """
    R = labels == 4
    d_lv = ndi.distance_transform_edt(labels != 1)
    d_la = ndi.distance_transform_edt(labels != 2)
    d_ao = ndi.distance_transform_edt(labels != 3)
    zz, yy, xx = np.indices(labels.shape, sparse=True)
    s = ((zz - apex[0]) * u[0] + (yy - apex[1]) * u[1] + (xx - apex[2]) * u[2]) / L
    ao_c, la_c = ndi.center_of_mass(labels == 3), la_center
    z_svc = la_c[0] + SVC_ABOVE_LA_MM
    # SVC: right of the ascending aorta, above the junction, not behind it (that is the right PA)
    svc = R & (zz >= z_svc) & (xx < ao_c[2]) & (yy < ao_c[1] + 5) & (d_ao > 1) & (d_ao <= 25)
    lab_s, n = ndi.label(svc)
    if n:
        sizes = ndi.sum(svc, lab_s, range(1, n + 1))
        svc = lab_s == (np.argmax(sizes) + 1)
    # LV myocardium (shell around the LV at ventricular level) and thin LA / aortic walls stay label 4
    wall = R & ~svc & (((d_lv <= LV_WALL_MM) & (s <= 1.05)) | (d_la <= THIN_WALL_MM) | (d_ao <= THIN_WALL_MM))
    Q = ndi.binary_opening(R & ~wall & ~svc, ball(2))
    lab_q, n = ndi.label(Q)
    if n:
        sizes = ndi.sum(Q, lab_q, range(1, n + 1))
        Q = np.isin(lab_q, np.flatnonzero(sizes >= RH_MIN_ML * 1000) + 1)
    if not Q.any():
        log("  right heart: nothing left to split"); labels[svc] = 8; return
    markers = np.zeros(labels.shape, np.int32)
    top = np.where(Q.any((1, 2)))[0].max()
    for z in range(max(top - PA_TOP_MM, 0), top + 1):     # PA: top slices, left of the aorta
        l2, n2 = ndi.label(Q[z])
        for i in range(1, n2 + 1):
            if ndi.center_of_mass(l2 == i)[1] > ao_c[2]:
                markers[z][l2 == i] = 7
    mid = Q & (s > 0.35) & (s < 0.65)                       # RV: mid-ventricle, farthest from the LV
    if mid.any():
        markers[mid & (d_lv >= np.percentile(d_lv[mid], 80))] = 6
    d_q = ndi.distance_transform_edt(Q)                     # RA: atrial level, patient-right of LA and aorta
    ra = Q & (s > 1.1) & (zz < z_svc - 6) & (xx < min(la_c[2], ao_c[2]) - 10) & (markers == 0)
    if ra.any():
        markers[ra & (d_q >= np.percentile(d_q[ra], 70))] = 5
    ws = watershed(-ndi.gaussian_filter(d_q, 1.5), markers, mask=Q)   # boundaries at the narrowings (valves, junctions)
    labels[ws > 0] = ws[ws > 0]
    labels[svc] = 8
    log("  right heart: " + ", ".join(f"{LABELS[k][0]} {(labels == k).sum() / 1000:.1f}" for k in (5, 6, 7, 8)) + " mL")


def save_overlay(hu, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([(0, 0, 0, 0)] + [LABELS[i][1][:3] + (0.45,) for i in sorted(LABELS)])
    Z, Y, X = hu.shape
    zs = np.where(labels.any((1, 2)))[0]
    zq = np.linspace(zs.min(), zs.max(), 6)[1:-1].astype(int)
    ys = int(np.mean(np.where(labels > 0)[1])); xs = int(np.mean(np.where(labels > 0)[2]))
    views = [("Axial z=%d" % z, hu[z], labels[z]) for z in zq]
    views += [("Coronal", hu[:, ys][::-1], labels[:, ys][::-1]),
              ("Sagittal", hu[:, :, xs][::-1], labels[:, :, xs][::-1])]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor="#111")
    for ax, (t, im, lb) in zip(axs.flat, views):
        ax.imshow(np.clip(im, -200, 600), cmap="gray")
        ax.imshow(lb, cmap=cmap, vmin=0, vmax=len(LABELS), interpolation="nearest")
        ax.set_title(t, color="w"); ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABELS[i][1][:3]) for i in sorted(LABELS)]
    fig.legend(handles, [LABELS[i][0] for i in sorted(LABELS)], loc="lower center",
               ncol=4, facecolor="#222", labelcolor="w")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(path, dpi=80, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(series=hio.DEFAULT_SERIES, spacing=1.0, log=print):
    t0 = time.time()
    os.makedirs(hio.OUT_MODELS, exist_ok=True); os.makedirs(hio.OUT_VIS, exist_ok=True)
    folder = os.path.join(hio.DATASET, series)
    log(f"[1/4] Loading DICOM: {folder}")
    img = hio.resample_iso(hio.load_series(folder), spacing)
    hu = sitk.GetArrayFromImage(img)
    log(f"      volume {hu.shape} @ {spacing} mm")
    log("[2/4] Segmenting heart ...")
    labels = segment(hu, log)
    log("[3/4] Saving NIfTI ...")
    lab_img = sitk.GetImageFromArray(labels); lab_img.CopyInformation(img)
    sitk.WriteImage(lab_img, os.path.join(hio.OUT_MODELS, "heart_labels.nii.gz"))
    sitk.WriteImage(img, os.path.join(hio.OUT_MODELS, "heart_ct_1mm.nii.gz"))
    vox_ml = spacing ** 3 / 1000.0
    stats = {"series": series, "spacing_mm": spacing, "shape_zyx": list(hu.shape),
             "labels": {LABELS[i][0]: round(float((labels == i).sum() * vox_ml), 1) for i in LABELS},
             "whole_heart_ml": round(float((labels > 0).sum() * vox_ml), 1),
             # LV long axis in patient LPS mm (used by build_heart_model.py for motion)
             "lv_long_axis_lps_mm": {k: list(img.TransformContinuousIndexToPhysicalPoint(v[::-1]))
                                     for k, v in segment.long_axis.items()}}
    with open(os.path.join(hio.OUT_MODELS, "segmentation_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log("[4/4] Saving overlay figure ...")
    save_overlay(hu, labels, os.path.join(hio.OUT_VIS, "segmentation_overlay.png"))
    log(f"Done in {time.time() - t0:.1f}s  -> volumes (mL): {stats['labels']}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=hio.DEFAULT_SERIES)
    ap.add_argument("--spacing", type=float, default=1.0)
    a = ap.parse_args()
    main(a.series, a.spacing)
