"""
Tests for the XR case package (case_package.py, CardioTwin XR roadmap step 11).

Fast: the model -> voxel transforms agree with SimpleITK (glTF, and Unity's mirrored X), and the raw volumes
are written x fastest. With the patient outputs: a package built in a temporary folder lines up with its
model (mesh vertices just inside each surface land in their own label, half as far inside for the thin
coronary arteries; contrast blood in the LV), lists
every mesh of the model, carries no patient identifiers or scan dates, and has a light model for headsets
(same structures and centre, about a quarter of the triangles).
"""
import os, re, shutil

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import build_heart_model as bhm
import case_package as cp
import heart_io as hio

INPUTS = ["heart_labels.nii.gz", "heart_ct_1mm.nii.gz", "segmentation_stats.json"]
OPTIONAL = [f for f in cp.MODEL_INPUTS if f not in INPUTS] + ["calcium_stats.json"]    # vessels, fat, DL chambers, coronaries
THIN = {bhm.COR_NAME: 0.00075}      # inset (m) for structures thinner than 3 mm (others 1.5 mm)
HAS_OUTPUTS = all(os.path.exists(os.path.join(hio.OUT_MODELS, f)) for f in INPUTS)
IDENTIFYING = ("patient", "birth", "accession", "institution", "physician", "operator", "study", "uid", "date")


def _img():
    img = sitk.Image(20, 30, 40, sitk.sitkUInt8)
    img.SetSpacing((0.7, 1.2, 2.5))
    img.SetOrigin((-50.0, 20.0, -110.0))
    return img


def test_model_to_voxel_matches_simpleitk():
    img, origin = _img(), np.array([-40.0, 35.0, -60.0])
    lps = np.random.default_rng(0).uniform([-50, 20, -110], [-36, 56, 0], (50, 3))
    model = bhm.lps_to_gltf(lps) - bhm.lps_to_gltf(origin)
    want = np.array([img.TransformPhysicalPointToContinuousIndex(p.tolist()) for p in lps])
    M, U = cp.voxel_from_model(img, origin), cp.voxel_from_model(img, origin, unity=True)
    assert np.allclose(model @ M[:3, :3].T + M[:3, 3], want)
    assert np.allclose((model * [-1, 1, 1]) @ U[:3, :3].T + U[:3, 3], want)    # glTFast mirrors X
    idx = np.array([[3, 7, 11]])                                                 # (z, y, x)
    assert np.allclose(cp.model_from_index(img, origin, idx) @ M[:3, :3].T + M[:3, 3], idx[:, ::-1])


def test_raw_volumes_are_x_fastest_little_endian(tmp_path):
    a = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4) * 1000            # (z, y, x)
    cp.write_raw(tmp_path / "v.raw", a, np.uint16)
    b = (tmp_path / "v.raw").read_bytes()
    z, y, x = 1, 2, 3
    i = x + 4 * (y + 3 * z)
    assert len(b) == a.size * 2 and int.from_bytes(b[2 * i:2 * i + 2], "little") == a[z, y, x]


def _walk(d):
    """(keys, string values) of a JSON document, recursively."""
    keys, strings = [], []
    stack = [d]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            keys += list(x)
            stack += list(x.values())
        elif isinstance(x, list):
            stack += x
        elif isinstance(x, str):
            strings.append(x)
    return keys, strings


@pytest.mark.skipif(not HAS_OUTPUTS, reason="needs the segment_heart.py outputs in output/3d_models")
def test_package_lines_up_with_its_model(tmp_path, monkeypatch):
    for f in INPUTS + OPTIONAL:
        if os.path.exists(os.path.join(hio.OUT_MODELS, f)):
            shutil.copy(os.path.join(hio.OUT_MODELS, f), tmp_path)
    monkeypatch.setattr(hio, "OUT_MODELS", str(tmp_path))
    monkeypatch.setattr(hio, "OUT_VIS", str(tmp_path))
    quiet = lambda *a, **k: None
    bhm.main(log=quiet)
    case = cp.main("TEST-1", out_dir=str(tmp_path / "pkg"), log=quiet)
    d = tmp_path / "pkg" / "TEST-1"
    assert sorted(p.name for p in d.iterdir()) == ["case.json", "ct.raw", "heart.glb", "heart_light.glb", "labels.raw"]
    # no identifiers: no such keys, and the package's own creation date is the only date in it
    keys, strings = _walk(case)
    assert not [k for k in keys if any(i in k.lower() for i in IDENTIFYING)]
    assert [w for w in strings if re.search(r"\d{4}-\d{2}-\d{2}", w)] == [case["created"]]
    # every mesh is listed, and the CT and labels line up with the meshes
    meshes = sorted(m["name"] for m in cp.read_glb_json(d / "heart.glb")["meshes"])
    assert sorted(s["name"] for s in case["structures"]) == meshes
    light = cp.read_glb_json(d / "heart_light.glb")
    assert sorted(m["name"] for m in light["meshes"]) == meshes
    assert light["extras"]["origin_lps_mm"] == case["model"]["origin_lps_mm"]
    assert case["model"]["light_triangles"] == cp.triangles(light) < 0.35 * case["model"]["triangles"]
    nx, ny, nz = case["grid"]["dims"]
    lab = np.fromfile(d / "labels.raw", np.uint8).reshape(nz, ny, nx)
    ct = np.fromfile(d / "ct.raw", "<u2").reshape(nz, ny, nx).astype(int) + case["ct"]["hu_offset"]
    M = np.array(case["grid"]["voxel_from_gltf"]).reshape(4, 4)

    def look(vol, p):
        v = np.clip(np.rint(np.asarray(p) @ M[:3, :3].T + M[:3, 3]).astype(int), 0, [nx - 1, ny - 1, nz - 1])
        return vol[v[:, 2], v[:, 1], v[:, 0]]

    scene = trimesh.load(d / "heart.glb", process=False)
    for s in case["structures"]:
        m = next(g for n, g in scene.geometry.items() if n.startswith(s["name"]))
        inset = THIN.get(s["name"], 0.0015)
        assert (look(lab, m.vertices - m.vertex_normals * inset) == s["label"]).mean() > 0.85, s["name"]
        assert look(lab, [s["anchor_m"]])[0] == s["label"], s["name"]
    lv = next(s for s in case["structures"] if s["name"] == "LV blood pool")
    assert look(ct, [lv["anchor_m"]])[0] > 200                                   # contrast blood

