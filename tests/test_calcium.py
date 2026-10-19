"""
Tests for the Agatston calcium score (calcium_score.py).

The arithmetic tests are synthetic and always run. The insertion tests
place artificial calcifications into the patient's real calcium-score scan
(TC_PTID has no calcium: its report says 0 Agatston). A coronary-like
lesion embedded in epicardial fat must be suggested and scored exactly.
A valve-like lesion in the blood pool must not be suggested. They need the
dataset and eat_cac.nii.gz from eat_patient.py.
"""
import os

import numpy as np
import pytest

import calcium_score as cs
import heart_io as hio


# ── Agatston arithmetic ──────────────────────────────────────────────────
@pytest.mark.parametrize("hu,w", [(129, 0), (130, 1), (199, 1), (200, 2), (299, 2), (300, 3), (399, 3), (400, 4), (900, 4)])
def test_density_weight_bands(hu, w):
    assert cs.density_weight(hu) == w


def test_find_lesions_scores_area_times_weight_and_applies_criteria():
    px, z_mm = 0.5, 2.5
    hu = np.full((3, 60, 60), 40.0, np.float32)
    hu[1, 10:14, 10:14] = 350          # 16 px = 4 mm2, weight 3            -> counted
    hu[1, 30:32, 30:32] = 500          # 4 px = 1 mm2 (exactly the minimum) -> counted, weight 4
    hu[1, 40, 40:43] = 600             # 3 px = 0.75 mm2 < 1 mm2            -> ignored
    hu[1, 50:54, 50:54] = 129          # below the threshold                -> ignored
    eat = np.zeros_like(hu, bool); eat[:, 8:34, 8:34] = True     # fat around the first two lesions
    zone = np.ones_like(hu, bool)
    lesions, _ = cs.find_lesions(hu, px, z_mm, eat, zone)
    assert len(lesions) == 2 and all(l["suggested"] for l in lesions)
    scale = z_mm / cs.AGATSTON_REF_MM
    assert sorted(l["agatston"] for l in lesions) == pytest.approx(sorted([4 * 3 * scale, 1 * 4 * scale]), abs=0.01)  # stored to 2 dp
    assert cs.score(lesions)[0] == pytest.approx(round((12 + 4) * scale, 1))
    assert cs.score(lesions, confirmed=set())[0] == 0              # all rejected by the reader


def test_cac_drs_categories():
    assert [cs.cac_drs(s) for s in (0, 1, 99.9, 100, 299, 300)] == [
        "0 (none)", "1 (mild, 1-99)", "1 (mild, 1-99)", "2 (moderate, 100-299)", "2 (moderate, 100-299)", "3 (severe, >= 300)"]


# ── insertion into the real calcium-score scan ────────────────────────────
EAT_CAC = os.path.join(hio.OUT_MODELS, "eat_cac.nii.gz")
HAS_SCAN = os.path.exists(EAT_CAC) and os.path.isdir(os.path.join(hio.DATASET, cs.CAC_SERIES))


@pytest.fixture(scope="module")
def scan():
    import SimpleITK as sitk
    import eat_segment as es
    img = hio.load_series(os.path.join(hio.DATASET, cs.CAC_SERIES))
    hu = np.maximum(sitk.GetArrayFromImage(img), -1024)[::-1].copy()
    eat = (sitk.GetArrayFromImage(sitk.ReadImage(EAT_CAC)) > 0)[::-1].copy()
    px, z_mm = img.GetSpacing()[0], img.GetSpacing()[2]
    heart, _ = es.heart_mask(hu, px, z_mm)
    _, zone, _ = es.outlines(heart, px, z_mm)
    return hu, eat, heart, zone, px, z_mm


def insert_disk(hu, k, cy, cx, r_px, value, slices=2):
    yy, xx = np.ogrid[:hu.shape[1], :hu.shape[2]]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_px ** 2
    for kk in range(k, k + slices):
        hu[kk][disk] = value
    return int(disk.sum())


@pytest.mark.skipif(not HAS_SCAN, reason="needs CT CALCIUM SCORE and eat_cac.nii.gz (run eat_patient.py)")
def test_patient_scan_has_only_small_faint_suggestions(scan):
    hu, eat, _, zone, px, z_mm = scan
    lesions, _ = cs.find_lesions(hu, px, z_mm, eat, zone)
    sug = [l for l in lesions if l["suggested"]]
    # report says 0 Agatston: whatever is suggested must be tiny and faint (the reader rejects it)
    assert all(l["area_mm2"] < 10 and l["peak_hu"] < 250 for l in sug)


@pytest.mark.skipif(not HAS_SCAN, reason="needs CT CALCIUM SCORE and eat_cac.nii.gz (run eat_patient.py)")
def test_inserted_coronary_lesion_is_found_and_valve_lesion_is_not(scan):
    from scipy import ndimage as ndi
    hu0, eat, heart, zone, px, z_mm = scan
    hu = hu0.copy()
    k = int(np.argmax(eat.sum((1, 2))))                          # the slice with the most epicardial fat
    thick = ndi.distance_transform_edt(eat[k]) * px              # coronary site: well inside a fat groove
    cy, cx = np.unravel_index(np.argmax(thick), thick.shape)
    n_cor = insert_disk(hu, k, cy, cx, r_px=1.3 / px, value=450)  # ~5 mm2 per slice, 2 slices, weight 4
    inside = ndi.distance_transform_edt(heart[k]) * px           # valve-like: deep in the heart, far from fat
    inside[eat[k]] = 0
    vy, vx = np.unravel_index(np.argmax(inside), inside.shape)
    insert_disk(hu, k, vy, vx, r_px=1.3 / px, value=450)
    lesions, lab3 = cs.find_lesions(hu, px, z_mm, eat, zone)
    cor = [l for l in lesions if lab3[k, cy, cx] == l["id"]]
    valve = [l for l in lesions if lab3[k, vy, vx] == l["id"]]
    assert len(cor) == 1 and cor[0]["suggested"], "coronary-like lesion in the fat was not suggested"
    assert cor[0]["agatston"] == pytest.approx(round(2 * n_cor * px * px * 4 * z_mm / cs.AGATSTON_REF_MM, 2), abs=0.02)
    assert len(valve) == 1 and not valve[0]["suggested"], "blood-pool (valve-like) lesion was suggested"
