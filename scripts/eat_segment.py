"""
==============================================================
  EAT - epicardial adipose tissue segmentation (non-contrast CT)
  UTM-Claude Research Project | rohayanti@utm.my

  EAT = fat inside the pericardium. Two stages:
    1  Rules (training-free) find the heart as the largest central
       soft-tissue blob and outline it per slice (convex hull). Near
       the apex/base, where the heart mask thins out, the outline is
       borrowed from neighbouring slices. Widened by CANDIDATE_MM, this
       zone holds ~100 % of the EAT on the benchmark, plus mediastinal,
       paracardial and abdominal fat.
    2  A gradient-boosted classifier (scikit-learn) scores every fat
       voxel in the zone from FEATURES: distances to the heart outline,
       lung and bone, local density (liver vs blood), position around
       the heart and slice level. Scores >= THRESHOLD are EAT.

  Rules only (no model) = fat within RULES_MARGIN_MM of the outline:
  the baseline the classifier is compared against.

  Trained and validated on the Rodriguez benchmark: eat_benchmark.py.
==============================================================
"""
import os

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

import heart_io as hio

# ── Tunable parameters (HU, mm) ───────────────────────────────────────────
FAT_RANGE = (-190, -30)     # HU  fat window for new scans (Rodriguez annotations used -200..-30)
SOFT_RANGE = (-30, 250)     # HU  soft tissue (blood, myocardium) on non-contrast CT
MEDIAN_MM = 1.75            # in-plane median filter before thresholds
HEART_ERODE_MM = 8.0        # opening that cuts the heart free from aorta, chest wall, liver
MIN_HEART_MM2 = 37.0        # smaller heart cross-sections get no outline of their own
NEIGHBOUR_MM = 6.0          # outline borrowed from slices this close (apex / base)
CANDIDATE_MM = 15.0         # candidate zone = outline widened by this much
LUNG_HU, BONE_HU = -500, 300
RULES_MARGIN_MM = 6.0       # rules-only EAT: fat within this distance of the outline
SMOOTH_MM = 0.0             # in-plane smoothing of scores before thresholding (0 = off; 1-3 mm lowered Dice)
THRESHOLD = 0.4             # classifier score for EAT (0.4 gave unbiased volumes; 0.5 under-segments ~8 mL)
MODEL_PATH = os.path.join(hio.PROJECT, "models", "eat_classifier.joblib")

FEATURES = ["hu", "hu_mean_3mm", "hu_mean_8mm", "fat_frac_3mm", "fat_frac_8mm", "dist_outline", "dist_heart",
            "dist_lung", "dist_bone", "level_rel", "mm_above_heart_bottom", "angle_sin", "angle_cos",
            "radius_rel", "slice_heart_hu_delta", "touches_heart", "fat_blob_mm2", "dist_zone_outline",
            "slice_heart_area_rel"]

_DS = 4  # in-plane downsampling for the heart search


def heart_mask(hu, px, z_mm):
    """Heart = largest central soft-tissue blob after an HEART_ERODE_MM opening. hu: (Z,Y,X), z = slice order."""
    k = 2 * int(round(MEDIAN_MM / px / 2)) + 1
    hs = ndi.median_filter(hu, size=(1, k, k))
    soft = (hs > SOFT_RANGE[0]) & (hs < SOFT_RANGE[1])
    small = soft[:, ::_DS, ::_DS]; samp = (z_mm, px * _DS, px * _DS)
    core = ndi.distance_transform_edt(small, sampling=samp) > HEART_ERODE_MM
    lab, n = ndi.label(core)
    if n == 0:
        return soft, hs
    Y, X = small.shape[1:]
    sizes = ndi.sum(core, lab, range(1, n + 1)); best, best_score = 1, -1
    for i in np.argsort(sizes)[::-1][:5]:          # big and close to the image centre
        com = ndi.center_of_mass(lab == i + 1)
        score = sizes[i] / (1 + np.hypot(com[1] - Y / 2, com[2] - X / 2) * px * _DS / 20)
        if score > best_score:
            best, best_score = i + 1, score
    grown = ndi.distance_transform_edt(lab != best, sampling=samp) <= HEART_ERODE_MM
    up = np.repeat(np.repeat(grown, _DS, 1), _DS, 2)[:, :hu.shape[1], :hu.shape[2]]
    return up & soft, hs


def outlines(heart, px, z_mm):
    """Per-slice convex hull of the heart, and the same borrowed from neighbouring slices."""
    Z = len(heart); area = heart.reshape(Z, -1).sum(1)
    own = np.stack([convex_hull_image(heart[k]) if area[k] * px * px > MIN_HEART_MM2 else np.zeros(heart[k].shape, bool)
                    for k in range(Z)])
    nb = int(round(NEIGHBOUR_MM / z_mm))
    zone = np.stack([own[max(0, k - nb):k + nb + 1].any(0) for k in range(Z)])
    return own, zone, area


def features(hu, px, z_mm, fat=None, fat_range=FAT_RANGE):
    """Candidate fat voxels and their FEATURES. Returns (X, idx, info); idx = (k, y, x) per row."""
    fat = ((hu >= fat_range[0]) & (hu <= fat_range[1])) if fat is None else fat
    heart, hs = heart_mask(hu, px, z_mm)
    own, zone, area = outlines(heart, px, z_mm)
    lung, bone = hs < LUNG_HU, hs > BONE_HU
    fat_hu = (hu >= fat_range[0]) & (hu <= fat_range[1])
    has = np.where(own.reshape(len(hu), -1).any(1))[0]
    z_top, z_bot = (has.min(), has.max()) if len(has) else (0, len(hu) - 1)
    mid = has[len(has) // 4: 3 * len(has) // 4] if len(has) >= 4 else has
    ref_hu = np.median(np.concatenate([hs[k][heart[k]] for k in mid])) if len(mid) else 40.0
    amax = max(area.max(), 1)
    far = lambda m: ndi.distance_transform_edt(~m) * px if m.any() else np.full(m.shape, 99.0)
    X, idx = [], []
    for k in range(len(hu)):
        if not zone[k].any():
            continue
        d_zone = ndi.distance_transform_edt(~zone[k]) * px
        cand = (d_zone <= CANDIDATE_MM) & fat[k]
        if not cand.any():
            continue
        hull = own[k] if own[k].any() else zone[k]
        d_hull = np.where(hull, -ndi.distance_transform_edt(hull) * px, ndi.distance_transform_edt(~hull) * px)
        ff = fat_hu[k].astype(np.float32)
        cy, cx = ndi.center_of_mass(hull); r_eq = np.sqrt(hull.sum() / np.pi)
        yy, xx = np.nonzero(cand); ang = np.arctan2(yy - cy, xx - cx)
        lab, n = ndi.label(cand)                                     # fat blobs; do they touch the heart?
        near = ndi.binary_dilation(heart[k], iterations=2) & cand
        ids = np.unique(lab[near]); touch = np.isin(lab, ids[ids > 0])
        blob_mm2 = ndi.sum(np.ones_like(lab), lab, np.arange(n + 1)) * px * px
        hu_delta = np.median(hs[k][heart[k]]) - ref_hu if heart[k].sum() > 50 else 0.0
        const = lambda v: np.full(len(yy), v, np.float32)
        f = np.stack([hu[k][cand], ndi.gaussian_filter(hs[k], 3 / px)[cand], ndi.gaussian_filter(hs[k], 8 / px)[cand],
                      ndi.gaussian_filter(ff, 3 / px)[cand], ndi.gaussian_filter(ff, 8 / px)[cand],
                      d_hull[cand], far(heart[k])[cand], far(lung[k])[cand], far(bone[k])[cand],
                      const((k - z_top) / max(z_bot - z_top, 1)), const((z_bot - k) * z_mm),
                      np.sin(ang), np.cos(ang), np.hypot(yy - cy, xx - cx) / r_eq, const(hu_delta),
                      touch[cand], blob_mm2[lab[cand]], d_zone[cand], const(area[k] / amax)], 1)
        X.append(f.astype(np.float32)); idx.append(np.c_[np.full(len(yy), k), yy, xx])
    X = np.concatenate(X) if X else np.zeros((0, len(FEATURES)), np.float32)
    idx = np.concatenate(idx).astype(np.int32) if idx else np.zeros((0, 3), np.int32)
    return X, idx, dict(heart=heart, outline=own, zone=zone)


def smooth_scores(shape, idx, score, px, sigma_mm=SMOOTH_MM):
    """Average each candidate's score with nearby candidates (Gaussian, in-plane, non-candidates ignored)."""
    if sigma_mm <= 0:
        return score
    s = np.zeros(shape, np.float32); w = np.zeros(shape, np.float32)
    s[tuple(idx.T)] = score; w[tuple(idx.T)] = 1.0
    sig = (0, sigma_mm / px, sigma_mm / px)
    return (ndi.gaussian_filter(s, sig) / np.maximum(ndi.gaussian_filter(w, sig), 1e-6))[tuple(idx.T)]


def load_model(path=MODEL_PATH):
    import joblib
    return joblib.load(path)


def segment_eat(hu, px, z_mm, model=None, fat=None, fat_range=FAT_RANGE, threshold=None):
    """EAT mask for a CT volume. hu: (Z,Y,X) HU; px: in-plane mm; z_mm: slice spacing.

    model: dict from load_model() (or None for the rules-only baseline).
    fat: optional boolean fat mask to use instead of the HU window (the
    benchmark passes the annotators' own fat pixels).
    threshold: defaults to the model's validated value (else THRESHOLD).
    Returns dict: eat (bool volume), score (float volume, NaN outside the
    candidates), heart, outline, zone.
    """
    X, idx, info = features(hu, px, z_mm, fat, fat_range)
    if model is None:
        score = (X[:, FEATURES.index("dist_outline")] <= RULES_MARGIN_MM).astype(np.float32)
        threshold = 0.5
    else:
        score = model["classifier"].predict_proba(X)[:, 1].astype(np.float32)
        score = smooth_scores(hu.shape, idx, score, px, model.get("smooth_mm", SMOOTH_MM))
        threshold = model.get("threshold", THRESHOLD) if threshold is None else threshold
    vol = np.full(hu.shape, np.nan, np.float32); vol[tuple(idx.T)] = score
    eat = np.zeros(hu.shape, bool); eat[tuple(idx.T)] = score >= threshold
    return dict(eat=eat, score=vol, **info)
