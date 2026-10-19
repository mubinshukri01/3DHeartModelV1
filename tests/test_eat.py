"""
Tests for EAT segmentation (eat_segment.py) and the Rodriguez benchmark loader (rodriguez.py).

The unit tests use synthetic data and always run. The last test needs the
Rodriguez dataset and the trained model (models/eat_classifier.joblib).
The full leave-one-patient-out benchmark (~45 min) is not part of the
suite: run scripts/eat_benchmark.py for that.
"""
import os

import numpy as np
import pytest

import eat_segment as es
import rodriguez as rz


# ── rodriguez.py ──────────────────────────────────────────────────────────
def test_decode_mask_classes():
    # grey = unlabelled fat, then red / green / blue-dominant pixels as the dataset stores them
    rgb = np.array([[[120, 120, 120], [173, 0, 102], [0, 150, 88], [98, 100, 144], [0, 0, 0]]], np.uint8)
    eat, ignore = rz.decode_mask(rgb)
    assert eat.tolist() == [[False, True, False, False, False]]
    assert ignore.tolist() == [[False, False, False, True, False]]


def test_fat_image_matches_dataset_mapping():
    hu = np.array([-250.0, -200.0, -115.0, -30.0, 0.0])
    np.testing.assert_allclose(rz.fat_image(hu), [0, 0, 127.5, 255, 0])


def test_register_recovers_known_scale_and_shift():
    rng = np.random.default_rng(0)
    from scipy import ndimage as ndi
    field = ndi.gaussian_filter(rng.normal(size=(512, 512)), 6)
    hu = (field / field.std() * 150 - 100).astype(np.float32)      # plenty of structure in the fat window
    px = 0.302734375                                                # a Siemens pixel size -> scale 0.8611
    s_true, ty, tx = round(px / rz.BMP_PX, 4), -7.25, 51.75
    bmp = rz.fat_image(rz.to_grid(hu, s_true, ty, tx))
    s, ty_est, tx_est, ncc = rz.register(hu, bmp, px)
    assert s == s_true and ncc > 0.99
    assert abs(ty_est - ty) <= 0.25 and abs(tx_est - tx) <= 0.25


# ── eat_segment.py ────────────────────────────────────────────────────────
def test_smooth_scores_keeps_constant_field_and_is_off_at_zero():
    idx = np.array([[0, y, x] for y in range(10, 20) for x in range(10, 20)])
    score = np.full(len(idx), 0.7, np.float32)
    np.testing.assert_allclose(es.smooth_scores((1, 32, 32), idx, score, 0.5, sigma_mm=2.0), 0.7, rtol=1e-5)
    assert es.smooth_scores((1, 32, 32), idx, score, 0.5, sigma_mm=0) is score


def phantom(px=0.7, z_mm=3.0, heart_r=45.0):
    """Lung background (-800 HU) holding a soft-tissue 'heart' ball (40 HU) wrapped in a
    3 mm fat shell (EAT, -100 HU) and a 2 mm pericardium (30 HU), plus a separate fat pad
    20-32 mm outside the heart (mediastinal fat, must not be called EAT)."""
    Z, N = 36, 256
    z, y, x = np.mgrid[:Z, :N, :N].astype(np.float32)
    dz, dy, dx = (z - Z / 2) * z_mm, (y - N / 2) * px, (x - N / 2) * px
    r = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
    hu = np.full((Z, N, N), -800.0, np.float32)
    hu[r < heart_r + 5] = 30                                  # pericardium (outermost layer)
    hu[r < heart_r + 3] = -100                                # EAT shell
    hu[r < heart_r] = 40                                      # heart
    pad = (np.abs(dy) < 12) & (dx > heart_r + 20) & (dx < heart_r + 32) & (np.abs(dz) < 20)
    hu[pad] = -100
    eat_true = (r >= heart_r) & (r < heart_r + 3)
    return hu, px, z_mm, eat_true, pad


def test_rules_only_finds_the_eat_shell_but_not_the_distant_fat():
    hu, px, z_mm, eat_true, pad = phantom()
    out = es.segment_eat(hu, px, z_mm, model=None)
    eat = out["eat"]
    assert out["heart"].sum() > 0
    mid = slice(8, 28)                                   # away from the ball's poles
    recall = (eat & eat_true)[mid].sum() / eat_true[mid].sum()
    assert recall > 0.9, f"EAT shell recall {recall:.2f}"
    assert not (eat & pad).any(), "fat 20+ mm outside the heart was called EAT"


def test_features_have_the_documented_columns():
    hu, px, z_mm, _, _ = phantom()
    X, idx, info = es.features(hu, px, z_mm)
    assert X.shape[1] == len(es.FEATURES) and len(X) == len(idx) > 0
    assert np.isfinite(X).all()


def test_build_uses_eat_only_on_the_heart_grid_and_drops_specks(tmp_path, monkeypatch):
    import SimpleITK as sitk
    import build_heart_model as bhm
    import heart_io as hio
    monkeypatch.setattr(hio, "OUT_MODELS", str(tmp_path))
    lab = sitk.Image(40, 40, 20, sitk.sitkUInt8); lab.SetSpacing((1, 1, 1))
    assert bhm.load_eat(lab) is None                                   # no eat_labels yet
    eat = np.zeros((20, 40, 40), np.uint8); eat[5:15, 5:15, 5:15] = 1; eat[18, 35, 35] = 1   # a blob + a 1-voxel speck
    img = sitk.GetImageFromArray(eat); img.CopyInformation(lab)
    sitk.WriteImage(img, str(tmp_path / "eat_labels.nii.gz"))
    m = bhm.load_eat(lab)
    assert m.sum() == 1000 and not m[18, 35, 35]
    other = sitk.Image(40, 40, 20, sitk.sitkUInt8); other.SetSpacing((1, 1, 2))
    assert bhm.load_eat(other, log=lambda *a: None) is None           # different grid -> ignored


# ── trained model on real data (needs dataset + model file) ───────────────
HAS_MODEL = os.path.exists(es.MODEL_PATH) and os.path.isdir(os.path.join(rz.ROOT, "DICOM"))


@pytest.mark.skipif(not HAS_MODEL, reason="needs dataset/Rodriguez and models/eat_classifier.joblib")
def test_trained_model_segments_a_benchmark_patient():
    model = es.load_model()
    assert model["features"] == es.FEATURES, "model was trained with a different feature list"
    P = rz.load_patient("AEdu")                            # one of the training patients: a smoke test, not validation
    out = es.segment_eat(P["hu"], P["px"], P["z_mm"], model=model, fat=P["fat"], fat_range=rz.FAT_RANGE)
    pred = out["eat"] & ~P["ignore"]; gt = P["eat"] & ~P["ignore"]
    dice = 2 * (pred & gt).sum() / (pred.sum() + gt.sum())
    assert dice > 0.8, f"Dice {dice:.3f}"


# ── TC_PTID: EAT step + model build (needs dataset, model and the step-1 outputs) ──
import heart_io as _hio
_STEP1 = [os.path.join(_hio.OUT_MODELS, f) for f in ("heart_ct_1mm.nii.gz", "heart_labels.nii.gz", "segmentation_stats.json")]
HAS_PATIENT = (os.path.exists(es.MODEL_PATH) and all(os.path.exists(f) for f in _STEP1)
               and os.path.isdir(os.path.join(_hio.DATASET, "CT CALCIUM SCORE")))
REFERENCE_EAT_ML = 73.4      # TC_PTID, CT CALCIUM SCORE, current model


@pytest.mark.skipif(not HAS_PATIENT, reason="needs CT CALCIUM SCORE, the model and segment_heart.py outputs")
def test_patient_eat_step_and_model_with_fat(tmp_path, monkeypatch):
    import json, shutil, struct
    import SimpleITK as sitk
    import build_heart_model as bhm
    import eat_patient as ep
    for f in _STEP1:
        shutil.copy(f, tmp_path)
    monkeypatch.setattr(_hio, "OUT_MODELS", str(tmp_path)); monkeypatch.setattr(_hio, "OUT_VIS", str(tmp_path))
    quiet = lambda *a, **k: None
    s = ep.main(log=quiet)
    assert s["eat_volume_ml"] == pytest.approx(REFERENCE_EAT_ML, rel=0.02)
    assert -100 < s["eat_mean_hu"] < -60                      # typical EAT attenuation
    assert np.linalg.norm(s["registration_to_ccta"]["shift_mm"]) < 10
    lab, eat = sitk.ReadImage(str(tmp_path / "heart_labels.nii.gz")), sitk.ReadImage(str(tmp_path / "eat_labels.nii.gz"))
    assert eat.GetSize() == lab.GetSize() and np.allclose(eat.GetOrigin(), lab.GetOrigin())
    bhm.main(bpm=72, log=quiet)
    with open(tmp_path / "heart_animated.glb", "rb") as fh:
        fh.read(12); n, _ = struct.unpack("<II", fh.read(8)); g = json.loads(fh.read(n))
    fat = [m for m in g["meshes"] if m["name"] == bhm.EAT_NAME]
    assert len(g["meshes"]) == len(bhm.LABELS) + 1 and len(fat) == 1 and len(fat[0]["primitives"][0]["targets"]) == 3
    assert g["extras"]["eat"]["volume_ml"] == s["eat_volume_ml"]
    assert (tmp_path / "heart_eat.stl").is_file() and (tmp_path / "eat_overlay.png").is_file()
