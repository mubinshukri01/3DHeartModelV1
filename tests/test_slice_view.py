"""
Tests for the DICOM viewer's slice display (slice_view.py, step 14).

Synthetic tests (always run): the radiological orientation of axial,
coronal and sagittal planes, their patient coordinates and pixel aspect,
overlay labels lining up in every view, position sync, the crossing line,
compositing and the HTML payload. The real-data test (needs the dataset)
checks that an axial plane cut from the CCTA volume equals the slice read
from its own file, so MPR and the per-file view agree.
"""
import base64, glob, io, json, os, re

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from PIL import Image

import heart_io as hio
import overlay as ov
import slice_view as sv

SPACING, ORIGIN = (0.5, 0.6, 1.0), (-5.0, 10.0, -100.0)
MARK = (9, 0, 13)      # voxel (x, y, z): patient left (x up), anterior (y low), superior (z up)


def _volume():
    a = np.zeros((14, 12, 10), np.int16)            # (z, y, x)
    a[MARK[2], MARK[1], MARK[0]] = 1000
    a[3:6, 4:8, 2:5] = 300                           # a block for the label checks
    img = sitk.GetImageFromArray(a)
    img.SetSpacing(SPACING)
    img.SetOrigin(ORIGIN)
    return img


@pytest.mark.parametrize("view,index,shape,where,spacing", [
    ("Axial", MARK[2], (12, 10), (0, 9), (0.6, 0.5)),       # anterior at the top, patient left on the right
    ("Coronal", MARK[1], (14, 10), (0, 9), (1.0, 0.5)),     # superior at the top, patient left on the right
    ("Sagittal", MARK[0], (14, 12), (0, 0), (1.0, 0.6)),    # superior at the top, anterior on the left
])
def test_planes_are_shown_the_radiological_way(view, index, shape, where, spacing):
    vol = _volume()
    pl = sv.plane(vol, view, index)
    assert pl.hu.shape == shape and pl.spacing == pytest.approx(spacing)
    assert tuple(np.argwhere(pl.hu == 1000)[0]) == where
    r, c = where                                     # the readout's position is the marker's position
    assert pl.p0 + r * pl.dr + c * pl.dc == pytest.approx(vol.TransformIndexToPhysicalPoint(MARK))
    assert pl.coord == pytest.approx(vol.TransformIndexToPhysicalPoint(MARK)[pl.axis])


@pytest.mark.parametrize("view,index", [("Axial", 4), ("Coronal", 5), ("Sagittal", 3)])
def test_overlay_labels_line_up_in_every_view(view, index):
    vol = _volume()
    labels = sitk.Cast(vol > 200, sitk.sitkUInt8)
    pl = sv.plane(vol, view, index)
    lab = sv.labels_on(pl, ov.label_slice(labels, pl.ref))
    assert lab.any() and np.array_equal(lab > 0, pl.hu > 200)


def test_position_sync_and_crossing_lines():
    vol = _volume()
    y3 = ORIGIN[1] + 3 * SPACING[1]
    assert sv.nearest(vol, "Coronal", y3) == 3 and sv.nearest(vol, "Coronal", y3 + 0.2) == 3
    assert sv.nearest(vol, "Coronal", 500.0) is None
    assert sv.nearest_z([0.0, 2.5, 5.0], 2.4, 1.25) == 1 and sv.nearest_z([0.0, 2.5, 5.0], 20.0, 1.25) is None
    cor = sv.plane(vol, "Coronal", 5)
    assert sv.crossing(cor, 2, ORIGIN[2] + 13) == ("row", 0.0)                  # top axial slice: top row
    assert sv.crossing(cor, 2, ORIGIN[2]) == ("row", 13.0)
    assert sv.crossing(cor, 0, ORIGIN[0] + 9 * SPACING[0]) == ("col", 9.0)
    assert sv.crossing(cor, 1, ORIGIN[1]) is None                                # parallel plane
    assert sv.crossing(cor, 2, ORIGIN[2] + 50) is None                           # outside


def test_from_slice_takes_the_normal_from_the_geometry():
    img = sitk.Image(8, 6, 1, sitk.sitkInt16)
    img.SetDirection((0, 0, 1, 1, 0, 0, 0, 1, 0))       # columns along y, rows along z: a sagittal-type image
    pl = sv.from_slice(sitk.GetArrayFromImage(img)[0], img)
    assert pl.axis == 0
    axial = sv.from_slice(np.zeros((6, 8)), sitk.Image(8, 6, 1, sitk.sitkInt16))
    assert axial.axis == 2
    assert sv.from_slice(np.zeros((4, 4, 3), np.uint8), None).ref is None       # RGB capture: no geometry


def test_compose_windows_and_blends():
    hu = np.array([[-1000, 40, 1000]], np.int16)
    rgb = sv.compose(hu, 350, 40, "gray")
    assert rgb[0, 0].tolist() == [0, 0, 0] and rgb[0, 2].tolist() == [255, 255, 255] and 120 < rgb[0, 1, 0] < 135
    lab = np.array([[0, 1, 0]], np.uint8)
    rgb = sv.compose(hu, 350, 40, "gray", lab, {1: ("LV", (1.0, 0.0, 0.0))}, 1.0)
    assert rgb[0, 1].tolist() == [255, 0, 0]


def test_html_carries_image_hu_and_labels():
    pl = sv.plane(_volume(), "Coronal", 0)
    lab = (pl.hu > 500).astype(np.uint8)
    rgb = sv.compose(pl.hu, 350, 40, "gray")
    page = sv.html(rgb, pl, 400, title="t", labels=lab, names={1: "Marker"}, line=("row", 2.0))
    meta = json.loads(re.search(r"const M = (\{.*?\});\n", page).group(1))
    png = Image.open(io.BytesIO(base64.b64decode(meta["png"])))
    assert png.size == (pl.hu.shape[1], pl.hu.shape[0]) and (meta["rows"], meta["cols"]) == pl.hu.shape
    hu = np.frombuffer(base64.b64decode(meta["hu"]), "<i2").reshape(pl.hu.shape)
    assert np.array_equal(hu, pl.hu) and meta["names"] == {"1": "Marker"} and meta["line"] == ["row", 2.0]
    assert "height:400px" in page


# ── real data ────────────────────────────────────────────────────────────
CCTA = os.path.join(hio.DATASET, "CT CCTA DL")


@pytest.mark.skipif(not os.path.isdir(CCTA), reason="needs the CT CCTA DL series")
def test_axial_plane_from_the_volume_equals_the_slice_file():
    vol = sitk.Cast(hio.load_series(CCTA), sitk.sitkInt16)
    f = sorted(glob.glob(os.path.join(CCTA, "*.dcm")))[100]
    img = sitk.ReadImage(f)
    z = float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2])
    k = sv.nearest(vol, "Axial", z)
    pl, one = sv.plane(vol, "Axial", k), sv.from_slice(sitk.GetArrayFromImage(img)[0], img)
    assert pl.coord == pytest.approx(one.coord, abs=0.01)
    assert np.array_equal(pl.hu, np.rint(one.hu).astype(np.int16))
    assert pl.p0 == pytest.approx(one.p0, abs=0.01)
    cor = sv.plane(vol, "Coronal", vol.GetSize()[1] // 2)
    assert cor.flip and cor.hu.shape == (vol.GetSize()[2], vol.GetSize()[0])
