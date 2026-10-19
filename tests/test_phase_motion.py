"""
Tests for the measured-motion step (phase_motion.py).

All synthetic and fast: the procedural weights at a phase, the removal of
rigid motion, the glTF/LPS conversion, simulated contrast removal, and the
B-spline registration recovering a known contraction of a phantom heart.
"""
import numpy as np
import pytest
import SimpleITK as sitk

import build_heart_model as bhm
import phase_motion as pm


def test_phase_weights_follow_the_cycle_curves():
    ph, W = bhm.cycle_weights(40)
    for i in (0, 9, 18, 30):                                   # phases on the coarse grid
        assert pm.phase_weights(ph[i]) == pytest.approx(W[i], abs=1e-9)
    dw = pm.phase_weights(0.45) - pm.phase_weights(pm.REST_PHASE)
    assert dw[0] > 0.5 and dw[1] == 0 and dw[2] > 0          # 45 %: ventricles contracted, no atrial kick


def test_remove_rigid_keeps_only_the_non_rigid_part():
    rng = np.random.default_rng(0)
    p = rng.normal(0, 30, (500, 3))
    a = np.radians(7)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rigid = p @ R.T + [3, -2, 5] - p
    assert np.abs(pm.remove_rigid(p, rigid)).max() < 1e-9
    c = p.mean(0)
    q = c + [3, -2, 5] + 0.95 * (p - c) @ R.T                 # rotate, move, and contract by 5 %
    assert pm.remove_rigid(p, q - p) == pytest.approx(-0.05 * (p - c) @ R.T, abs=1e-9)   # only the contraction is left


def test_gltf_lps_round_trip():
    p = np.array([[10.0, -20.0, 30.0], [-5.0, 7.5, 0.0]])
    assert pm.gltf_to_lps(bhm.lps_to_gltf(p)) == pytest.approx(p)


def _phantom(n=96, sp=1.0):
    """Lung, a fat body, and a 'heart' (myocardium shell around a contrast blood pool), centred."""
    c = (n - 1) / 2 * sp
    z, y, x = np.meshgrid(*[np.arange(n) * sp] * 3, indexing="ij")
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + ((z - c) * 1.2) ** 2)          # slightly flattened in z
    a = np.full(r.shape, -800.0)
    a[r < 44] = -100
    a[r < 25] = 150
    a[r < 19] = 350
    a += np.random.default_rng(0).normal(0, 10, a.shape)
    img = sitk.GetImageFromArray(a.astype(np.float32))
    img.SetSpacing((sp,) * 3)
    heart = sitk.GetImageFromArray((r < 25).astype(np.uint8))
    heart.CopyInformation(img)
    return img, heart, c


def test_deformable_recovers_a_known_contraction():
    img, heart, c = _phantom()
    n = img.GetSize()[0]
    z, y, x = np.meshgrid(*[np.arange(n, dtype=float)] * 3, indexing="ij")
    rel = np.stack([x - c, y - c, z - c], -1)
    r = np.linalg.norm(rel, axis=-1)
    truth_arr = -0.08 * rel * np.exp(-np.clip(r - 25, 0, None) / 10)[..., None]   # 8 % contraction, 2 mm at the surface
    truth = sitk.GetImageFromArray(truth_arr, isVector=True)
    truth.CopyInformation(img)
    moving = pm.warp_by(img, truth)
    tx, _ = pm.deformable(img, moving, heart, sitk.Euler3DTransform(), log=lambda *_: None)
    # points on the heart surface (r = 25 mm) in every direction
    d = np.random.default_rng(1).normal(size=(300, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pts = c + 25 * d * np.array([1, 1, 1 / 1.2])
    est = pm.sample_field(pm.field_of(tx, img), pts)
    true = pm.sample_field(truth, pts)
    err = np.linalg.norm(est - true, axis=1)
    assert np.linalg.norm(true, axis=1).mean() > 1.5
    assert err.mean() < 0.6, f"mean error {err.mean():.2f} mm"


def test_remove_contrast_flattens_the_heart_only():
    img, heart, c = _phantom(48, 2.0)
    out = sitk.GetArrayFromImage(pm.remove_contrast(img, heart))
    a = sitk.GetArrayFromImage(img)
    centre = tuple(int(round(c / 2.0)) for _ in range(3))
    assert abs(out[centre] - pm.NONCONTRAST_HU) < 60 and a[centre] > 300    # blood pool: 350 -> ~45 HU
    assert out[0, 0, 0] == a[0, 0, 0]                                       # lung untouched
