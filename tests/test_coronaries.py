"""
Tests for the coronary arteries (coronaries.py, and their part in build_heart_model.py; roadmap step 7).

Synthetic and fast: a 3 mm-wide artery on a sub-millimetre grid is meshed where it lies, at its volume, and
keeps about its volume as a label on the 1 mm grid; the arteries move with the wall they lie on; the cleanup
drops specks and the crop keeps the heart with a margin.
"""
import numpy as np
import pytest
import SimpleITK as sitk

import build_heart_model as bhm
import coronaries

AXIS = {"apex": [30.0, 30.0, 6.0], "base": [30.0, 30.0, 66.0], "la_center": [30.0, 30.0, 80.0]}   # LPS mm


def _artery():
    """A 3 mm-wide artery along x on a 0.4 x 0.4 x 0.625 mm grid, and its volume (mL)."""
    z, y, x = np.mgrid[:64, :100, :100]
    tube = (np.hypot((y - 50) * 0.4, (z - 32) * 0.625) <= 1.5) & (x >= 10) & (x < 90)
    img = sitk.GetImageFromArray(tube.astype(np.uint8))
    img.SetSpacing((0.4, 0.4, 0.625))
    img.SetOrigin((5.0, 3.0, 2.0))
    return img, tube.sum() * 0.4 * 0.4 * 0.625 / 1000


def test_artery_is_meshed_at_native_resolution_and_kept_on_the_grid():
    cor, ml = _artery()
    m = bhm.coronary_mesh(cor, 10 ** 5)
    assert abs(m.volume) * 1e6 == pytest.approx(ml, rel=0.1)
    lps = bhm.gltf_to_lps(m.vertices)
    assert np.allclose(lps.mean(0), [5 + 49.5 * 0.4, 3 + 50 * 0.4, 2 + 32 * 0.625], atol=0.2)   # where it lies
    assert np.ptp(lps[:, 1]) == pytest.approx(3.0, abs=0.4)                                     # 3 mm wide
    grid = sitk.Image(50, 50, 50, sitk.sitkUInt8)                                              # 1 mm grid around it
    on = bhm.coronary_grid(cor, grid)
    assert on.sum() / 1000 == pytest.approx(ml, rel=0.2)     # +16 % here; thin branches drop out (patient: +3 %)
    assert on[:, 20:26, 20].any(axis=1).sum() >= 2                                             # and in one piece across


def test_coronaries_move_with_the_wall_they_lie_on():
    z, y, x = np.mgrid[:90, :60, :60]
    rho = np.hypot(x - 30, y - 30)
    wall = np.zeros(z.shape, np.uint8)
    wall[(rho <= 14) & (z >= 4) & (z <= 68)] = 4
    wall[(rho <= 8) & (z >= 8) & (z <= 64)] = 1
    wall[(rho <= 5) & (z > 68) & (z <= 85)] = 3                 # aorta on top
    with_cor = wall.copy()
    artery = (rho > 14) & (rho <= 16) & (abs(x - 30) <= 2) & (y > 30) & (z >= 20) & (z <= 50)
    with_cor[artery] = bhm.COR_LABEL
    as_wall = wall.copy()
    as_wall[artery] = 4                                        # what it lies on, as if it were part of it
    a = bhm.motion_context(sitk.GetImageFromArray(with_cor), with_cor, AXIS)
    b = bhm.motion_context(sitk.GetImageFromArray(as_wall), as_wall, AXIS)
    p = bhm.lps_to_gltf(np.array([[30.0, 45.0, 35.0], [30.0, 46.0, 25.0]]))    # in the artery (LPS = x, y, z)
    assert all(np.allclose(s, t, atol=1e-9) for s, t in zip(a["targets"](p), b["targets"](p)))
    assert np.linalg.norm(a["targets"](p)[0], axis=1).min() > 1e-3                            # and it does move


def test_clean_drops_specks_and_crop_keeps_the_heart():
    mask = np.zeros((20, 20, 20), bool)
    mask[2:12, 5, 5] = True                                    # an artery piece, 10 voxels of 0.5 mm: 1.25 mm3
    mask[15, 15, 15] = True                                    # a speck
    kept, n = coronaries.clean(mask, (0.5, 0.5, 0.5))
    assert n == 0 and not kept.any()                           # both below MIN_PIECE_MM3 (2 mm3)
    kept, n = coronaries.clean(mask, (1.0, 1.0, 1.0))
    assert n == 1 and kept[2:12, 5, 5].all() and not kept[15, 15, 15]
    ref = sitk.GetImageFromArray(np.pad(np.ones((4, 4, 4), np.uint8), 8))  # heart: voxels 8-11, 1 mm grid
    img = sitk.Image(100, 100, 100, sitk.sitkUInt8)
    img.SetSpacing((0.5, 0.5, 0.5))
    crop = coronaries.crop_to(img, ref, margin_mm=2.0)
    lo, hi = np.array(crop.GetOrigin()), np.array(crop.GetOrigin()) + (np.array(crop.GetSize()) - 1) * 0.5
    assert np.all(lo <= 6.0) and np.all(lo >= 5.0) and np.all(hi >= 13.0) and np.all(hi <= 14.0)
