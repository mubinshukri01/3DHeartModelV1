"""
Tests for the procedural motion of build_heart_model.py (CardioTwin XR roadmap, step 10).

Synthetic and fast: the LV cavity keeps the designed EF while the wall keeps
its volume; the AV-plane descent fades out at the atrial roof (lower) and the
arch (higher); neighbouring structures' motions blend continuously (no seams),
each keeps its own motion inside; the fat moves with what it lies on; the RV
field does not depend on which points are evaluated together.
"""
import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import build_heart_model as bhm

AXIS = {"apex": [30.0, 30.0, 6.0], "base": [30.0, 30.0, 66.0], "la_center": [30.0, 30.0, 80.0]}   # LPS mm


def _motion():
    return bhm.CardiacMotion(AXIS, lambda p: np.full(len(p), 1.0))


def _along_axis(mesh, z0_mm):
    """trimesh primitive (axis z, metres) -> along the LV axis (glTF +Y), its base at LPS height z0_mm."""
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))   # z -> y
    m.apply_translation(bhm.lps_to_gltf(np.array([30.0, 30.0, z0_mm])))
    return m


def _volume_change(mesh, d):
    return abs(trimesh.Trimesh(mesh.vertices + d, mesh.faces, process=False).volume) / abs(mesh.volume) - 1


def test_cavity_keeps_the_ef_and_the_wall_its_volume():
    mo = _motion()
    r = np.random.default_rng(0)
    pts = np.c_[r.uniform(-15, 15, (20000, 2)), r.uniform(0, 60, 20000)]
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= 15] + [30, 30, 6]                    # LV cavity: radius 15 mm
    mo.fit_cavity(bhm.lps_to_gltf(pts))
    # mid-ventricle (s 0.3-0.9: no apex taper, no AV-plane fade): cavity and a 10 mm wall around it
    cavity = _along_axis(trimesh.creation.cylinder(0.0149, 0.036, sections=96).apply_translation([0, 0, 0.018]), 24)
    wall = _along_axis(trimesh.creation.annulus(0.0151, 0.025, 0.036, sections=96).apply_translation([0, 0, 0.018]), 24)
    assert -_volume_change(cavity, mo.ventricular_systole(cavity.vertices)) == pytest.approx(0.60, abs=0.02)
    assert _volume_change(wall, mo.ventricular_systole(wall.vertices)) == pytest.approx(0.0, abs=0.02)


def test_atrial_kick_fills_the_cavity_and_keeps_the_wall():
    mo = _motion()
    r = np.random.default_rng(0)
    pts = np.c_[r.uniform(-15, 15, (20000, 2)), r.uniform(0, 60, 20000)]
    mo.fit_cavity(bhm.lps_to_gltf(pts[np.hypot(pts[:, 0], pts[:, 1]) <= 15] + [30, 30, 6]))
    cavity = _along_axis(trimesh.creation.cylinder(0.0149, 0.036, sections=96).apply_translation([0, 0, 0.018]), 24)
    wall = _along_axis(trimesh.creation.annulus(0.0151, 0.025, 0.036, sections=96).apply_translation([0, 0, 0.018]), 24)
    gain = (1 + mo.awave) ** 2 * 1.03 - 1                              # radius and length grow
    assert _volume_change(cavity, mo.atrial_systole(cavity.vertices)) == pytest.approx(gain, abs=0.02)
    assert _volume_change(wall, mo.atrial_systole(wall.vertices)) == pytest.approx(0.0, abs=0.02)


def test_lv_long_axis_from_the_labels():
    labels = np.zeros((70, 40, 40), np.uint8)
    z, y, x = np.mgrid[:70, :40, :40]
    labels[(np.hypot(x - 20, y - 20) <= 8) & (z >= 10) & (z <= 40)] = 1      # LV, apex at the bottom (z 10)
    labels[(np.hypot(x - 20, y - 20) <= 9) & (z > 40) & (z <= 55)] = 2      # LA on top of it
    axis = bhm.lv_long_axis(labels, sitk.GetImageFromArray(labels))        # 1 mm voxels at the origin: LPS = (x, y, z)
    assert axis["base"][2] == pytest.approx(39.5, abs=1.5) and axis["apex"][2] < 13
    assert axis["la_center"][2] == pytest.approx(48, abs=0.5)
    assert np.allclose(np.array(axis["apex"])[:2], 20, atol=3) and np.allclose(np.array(axis["base"])[:2], 20, atol=0.5)


def test_chamber_source(tmp_path, monkeypatch):
    import json
    import heart_io as hio
    monkeypatch.setattr(hio, "OUT_MODELS", str(tmp_path))
    img = sitk.GetImageFromArray(np.zeros((10, 10, 10), np.uint8))
    assert bhm.chamber_source(img) == "rules"
    with pytest.raises(FileNotFoundError):
        bhm.chamber_source(img, "dl")
    sitk.WriteImage(img, str(tmp_path / "heart_labels_dl.nii.gz"))
    (tmp_path / "dl_segmentation.json").write_text(json.dumps({"chambers": False}))
    assert bhm.chamber_source(img) == "rules"                                # open task only: no chambers
    (tmp_path / "dl_segmentation.json").write_text(json.dumps({"chambers": True}))
    assert bhm.chamber_source(img) == "dl" and bhm.chamber_source(img, "rules") == "rules"


def test_av_plane_descent_fades_out_at_the_roof_and_the_arch():
    mo = _motion()
    top = lambda mm: bhm.lps_to_gltf(np.array([[30.0, 30.0, 66.0 + mm]]))   # on the axis, mm above the mitral plane
    down = lambda mm, **kw: -(mo.ventricular_systole(top(mm), **kw) @ mo.u)[0] * 1000
    full = mo.ls * mo.L * 1000
    assert down(5) == pytest.approx(full, rel=0.1)                           # the floor follows the AV plane
    assert down(bhm.ANCHOR_MM + 1) == pytest.approx(0.0, abs=1e-9)           # the atrial roof is held
    assert 0.2 * full < down(bhm.ANCHOR_MM + 1, s_anchor=mo.s_anchor_artery) < full   # arteries: held higher up
    assert down(bhm.ARTERY_ANCHOR_MM + 1, s_anchor=mo.s_anchor_artery) == pytest.approx(0.0, abs=1e-9)


def _heart():
    """LV in a myocardium shell, an aorta on top, an IVC block beside it; fat on the wall and fat beyond the IVC."""
    z, y, x = np.mgrid[:90, :60, :80]
    rho = np.hypot(x - 30, y - 30)
    lab = np.zeros(z.shape, np.uint8)
    lab[(rho <= 14) & (z >= 4) & (z <= 68)] = 4
    lab[(rho <= 8) & (z >= 8) & (z <= 64)] = 1
    lab[(rho <= 5) & (z > 68) & (z <= 85)] = 3
    lab[(x > 44) & (x <= 56) & (abs(y - 30) <= 6) & (z >= 10) & (z <= 40)] = 9
    lab[(x > 56) & (x <= 60) & (abs(y - 30) <= 6) & (z >= 10) & (z <= 40)] = bhm.EAT_LABEL
    lab[(x >= 12) & (x < 16) & (abs(y - 30) <= 6) & (z >= 20) & (z <= 50)] = bhm.EAT_LABEL
    img = sitk.GetImageFromArray(lab)
    return lab, img, bhm.motion_context(img, lab, AXIS)


@pytest.fixture(scope="module")
def heart():
    return _heart()


def _gltf(xyz):
    return bhm.lps_to_gltf(np.asarray(xyz, float))


def _own(ctx, k, p):
    return bhm.motion_targets(ctx["motion"], k, p, ctx["ra_centre"], ctx["rv_centre"], ctx["aorta_center"],
                              ctx["pa_center"], ctx["pa_dist"], ctx["dao"])


def test_structures_keep_their_own_motion_inside_and_blend_across(heart):
    lab, img, ctx = heart
    lv, ivc = _gltf([[30, 30, 36]]), _gltf([[52, 30, 25]])
    for k, p in ((1, lv), (9, ivc)):
        assert all(np.allclose(a, b, atol=2e-5) for a, b in zip(ctx["targets"](p), _own(ctx, k, p)))   # < 0.02 mm
    # across the myocardium / IVC interface (x = 44.5): own motions jump there, the blend is continuous
    line = _gltf(np.c_[np.arange(38, 51, 0.25), np.full(52, 30.0), np.full(52, 25.0)])
    blended = ctx["targets"](line)[0] * 1000
    jump = np.linalg.norm(_own(ctx, 4, line[:1])[0] - _own(ctx, 9, line[:1])[0]) * 1000
    assert jump > 1.0
    assert np.linalg.norm(np.diff(blended, axis=0), axis=1).max() < jump / 4


def test_fat_moves_with_what_it_lies_on(heart):
    lab, img, ctx = heart
    on_wall, beyond_ivc = _gltf([[14, 30, 35]]), _gltf([[58, 30, 25]])
    assert np.linalg.norm(ctx["targets"](beyond_ivc)[0]) < 1e-5                       # the IVC is held
    wall = _own(ctx, 4, on_wall)[0]
    assert np.linalg.norm(ctx["targets"](on_wall)[0] - wall) < 0.05 * np.linalg.norm(wall)


def test_rv_field_does_not_depend_on_the_points_evaluated_together():
    mo = _motion()
    mo.rv_scale = 0.03
    c_rv = _gltf([45.0, 30.0, 30.0])
    p = _gltf(np.random.default_rng(1).uniform([20, 20, 10], [70, 40, 60], (500, 3)))
    assert np.allclose(mo.rv_systole(p, c_rv)[:50], mo.rv_systole(p[:50], c_rv))
