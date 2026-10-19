"""
Tests for the segmentation overlay of the DICOM viewer (overlay.py).

Synthetic tests (always run): labels land on the right pixels of a slice
with another pixel size and field of view, the same-scan rule, colours, and
the layer list. The real-data tests need the dataset and the pipeline's
output/3d_models: the heart overlay covers the same area (mm2) on the CCTA
and on the Lungs reconstruction of that scan, and is refused on the
calcium-score scan (another breath-hold), which gets the fat instead.
"""
import glob, json, os

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

import heart_io as hio
import overlay as ov


def _labels():
    """1 mm label volume with a 20 mm cube of label 3 at x, y, z = 10..29 mm."""
    a = np.zeros((40, 40, 40), np.uint8)
    a[10:30, 10:30, 10:30] = 3
    img = sitk.GetImageFromArray(a)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def _slice(spacing, origin, size=(80, 80)):
    img = sitk.Image(size[0], size[1], 1, sitk.sitkInt16)
    img.SetSpacing((spacing, spacing, 0.625))
    img.SetOrigin(origin)
    return img


@pytest.mark.parametrize("spacing", [0.377, 0.744])
def test_labels_land_on_the_same_millimetres_for_any_pixel_size(spacing):
    lab = ov.label_slice(_labels(), _slice(spacing, (0.0, 0.0, 20.0)))
    ys, xs = np.nonzero(lab == 3)
    assert set(np.unique(lab)) == {0, 3}
    assert xs.min() * spacing == pytest.approx(9.5, abs=spacing) and xs.max() * spacing == pytest.approx(29.5, abs=spacing)
    assert (lab == 3).sum() * spacing ** 2 == pytest.approx(400, rel=0.1)          # 20 x 20 mm
    assert not ov.label_slice(_labels(), _slice(spacing, (0.0, 0.0, 35.0))).any()  # above the cube


def _header(frame, time):
    ds = pydicom.Dataset()
    ds.FrameOfReferenceUID, ds.AcquisitionTime = frame, time
    return ds


def test_same_scan_needs_the_same_acquisition_not_just_the_same_frame():
    ccta = _header("1.2.3", "111849.320046")
    assert ov.same_scan(_header("1.2.3", "111849"), ccta)               # SSFreeze: time truncated
    assert not ov.same_scan(_header("1.2.3", "111207.230998"), ccta)    # calcium score, 6 min earlier
    assert not ov.same_scan(_header("9.9.9", "111849.320046"), ccta)    # another frame of reference
    assert not ov.same_scan(pydicom.Dataset(), ccta)                     # no tags


def test_rgba_and_legend():
    lab = np.array([[0, 1], [3, 3]], np.uint8)
    img = ov.rgba(lab, ov.HEART, 0.5)
    assert img[0, 0, 3] == 0 and img[0, 1, 3] == 0.5
    assert tuple(img[1, 0, :3]) == pytest.approx(ov.HEART[3][1])
    shown = ov.present(lab, ov.HEART, 2.0)
    assert [(n, a) for n, _, a in shown] == [("Aorta", 4.0), ("LV blood pool", 2.0)]


def test_layers_list_only_files_that_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(hio, "OUT_MODELS", str(tmp_path))
    (tmp_path / "segmentation_stats.json").write_text(json.dumps({"series": "CT CCTA DL"}))
    (tmp_path / "eat_stats.json").write_text(json.dumps({"series": "CT CALCIUM SCORE"}))
    for f in ("heart_labels.nii.gz", "eat_cac.nii.gz"):
        (tmp_path / f).write_bytes(b"")
    layers = ov.layers()
    assert list(layers) == ["Heart (rules)", "Epicardial fat"]            # no DL labels, no eat_labels
    assert [e[1] for e in layers["Epicardial fat"]] == ["CT CALCIUM SCORE"]


# ── real data ────────────────────────────────────────────────────────────
HAS_DATA = all(os.path.isdir(os.path.join(hio.DATASET, s)) for s in ("CT CCTA DL", "CT LUNGS 0.625MM", "CT CALCIUM SCORE")) \
    and os.path.exists(os.path.join(hio.OUT_MODELS, "heart_labels.nii.gz"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="needs the dataset and output/3d_models")


def _slice_at(series, z=None):
    """(header, SimpleITK slice) of a series: the middle slice, or the one nearest to height z (mm)."""
    files = sorted(glob.glob(os.path.join(hio.DATASET, series, "*.dcm")))
    heads = [(f, pydicom.dcmread(f, stop_before_pixels=True)) for f in files]
    f, ds = heads[len(heads) // 2] if z is None else min(heads, key=lambda h: abs(float(h[1].ImagePositionPatient[2]) - z))
    return ds, sitk.ReadImage(f)


@needs_data
def test_heart_overlay_matches_across_reconstructions_and_is_refused_on_another_scan():
    ov.series_header.cache_clear()
    heart = ov.layers()["Heart (rules)"]
    ds, img = _slice_at("CT CCTA DL")
    z = float(ds.ImagePositionPatient[2])
    area = {}
    for series in ("CT CCTA DL", "CT LUNGS 0.625MM"):
        ds_s, img_s = _slice_at(series, z)
        assert abs(float(ds_s.ImagePositionPatient[2]) - z) < 0.5
        entry = ov.layer_for(ds_s, heart)
        assert entry is not None, series
        lab = ov.label_slice(ov.load_labels(entry[0]), img_s)
        area[series] = (lab > 0).sum() * img_s.GetSpacing()[0] * img_s.GetSpacing()[1]
    assert area["CT CCTA DL"] > 1000                                            # a real heart section (> 10 cm2)
    assert area["CT LUNGS 0.625MM"] == pytest.approx(area["CT CCTA DL"], rel=0.05)
    ds_c, _ = _slice_at("CT CALCIUM SCORE")
    assert ov.layer_for(ds_c, heart) is None                                    # another breath-hold
    fat = ov.layers().get("Epicardial fat")
    if fat:
        assert ov.layer_for(ds_c, fat)[1] == "CT CALCIUM SCORE"                  # the fat measured on it
