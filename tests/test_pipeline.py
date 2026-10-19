r"""
==============================================================
  Regression tests for the 3D heart pipeline
  UTM-Claude Research Project | rohayanti@utm.my

  The full-pipeline tests run segment_heart + build_heart_model on
  CT CCTA DL, writing to a temporary folder so the real output/ is left
  alone. They compare the result with the reference run (volumes below,
  and the label map in tests/reference/). They need the DICOM dataset and
  are skipped without it. The unit tests at the bottom always run.

  Usage (from the 3DHeartModelV1 folder):
    .venv\Scripts\python -m pytest tests -v

  If you change the segmentation on purpose, check
  output/visualizations/segmentation_overlay.png first. Then update
  REFERENCE_ML and copy the new heart_labels.nii.gz into tests/reference/.
==============================================================
"""
import json, os, struct

import numpy as np
import pytest
import SimpleITK as sitk

import build_heart_model as bhm
import heart_io as hio
import segment_heart as seg

# ── reference run (CT CCTA DL, 1 mm, 72 bpm) ─────────────────────────────
REFERENCE_ML = {"LV blood pool": 33.0, "LA blood pool": 61.1, "Aorta": 23.7, "Myocardium": 102.0,
                "Right atrium": 23.3, "Right ventricle": 48.8, "Pulmonary artery": 29.9, "Superior vena cava": 3.6}
REFERENCE_WHOLE_HEART_ML = 325.4
# LV mesh volumes over the procedural cycle. Since 9 Oct 2026 the meshes share surfaces and match the
# voxel volumes; before, a Taubin sign bug shrank them (35.3 / 14.3 mL, EF 59.5 %). Since 10 Oct 2026 the LV
# uses its blended morph targets as in the GLB (before: 36.4 / 15.0 mL, EF 58.9 %). Since the atrial kick was balanced
# on the deep-learning chambers (12 Oct 2026; A-wave radius gain 5 -> 7 %) the rule-based EDV is 37.3 (was 36.4).
REFERENCE_LV_EDV_ML, REFERENCE_LV_ESV_ML, REFERENCE_EF = 37.3, 15.1, 59.4
REFERENCE_LABELS = os.path.join(os.path.dirname(__file__), "reference", "heart_labels.nii.gz")

# Tolerances absorb numeric noise from other library versions. A real change
# to the segmentation logic moves the result much further than this.
VOLUME_RTOL = 0.01   # 1 % per structure
MIN_DICE = 0.99      # voxel overlap with the reference label map

HAS_DATA = os.path.isdir(os.path.join(hio.DATASET, hio.DEFAULT_SERIES))
needs_data = pytest.mark.skipif(not HAS_DATA, reason=f"no {hio.DEFAULT_SERIES} series in {hio.DATASET}")


def read_glb_json(path):
    """Return the JSON chunk of a GLB file, checking the binary header on the way."""
    with open(path, "rb") as fh:
        magic, version, total = struct.unpack("<III", fh.read(12))
        assert magic == 0x46546C67 and version == 2, "not a glTF 2.0 binary"
        assert total == os.path.getsize(path), "GLB length field does not match file size"
        js_len, js_type = struct.unpack("<II", fh.read(8))
        assert js_type == 0x4E4F534A, "first GLB chunk is not JSON"
        return json.loads(fh.read(js_len))


def read_glb_bin(path):
    """Return the binary chunk of a GLB file."""
    with open(path, "rb") as fh:
        fh.seek(12)
        js_len, _ = struct.unpack("<II", fh.read(8))
        fh.seek(20 + js_len)
        bin_len, bin_type = struct.unpack("<II", fh.read(8))
        assert bin_type == 0x004E4942, "second GLB chunk is not BIN"
        return fh.read(bin_len)


def dice(a, b):
    return 2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run both pipeline steps once, into a temporary output folder."""
    out = tmp_path_factory.mktemp("pipeline")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hio, "OUT_MODELS", str(out / "3d_models"))
        mp.setattr(hio, "OUT_VIS", str(out / "visualizations"))
        quiet = lambda *a, **k: None
        stats = seg.main(log=quiet)
        info = bhm.main(bpm=72.0, log=quiet)
    return {"stats": stats, "info": info, "models": out / "3d_models", "vis": out / "visualizations"}


# ── full pipeline (needs the dataset) ────────────────────────────────────
@needs_data
def test_structure_volumes_match_reference(pipeline_run):
    labels = pipeline_run["stats"]["labels"]
    assert set(labels) == set(REFERENCE_ML)
    for name, ref in REFERENCE_ML.items():
        assert labels[name] == pytest.approx(ref, rel=VOLUME_RTOL), name
    assert pipeline_run["stats"]["whole_heart_ml"] == pytest.approx(REFERENCE_WHOLE_HEART_ML, rel=VOLUME_RTOL)


@needs_data
def test_label_map_overlaps_reference(pipeline_run):
    new = sitk.ReadImage(str(pipeline_run["models"] / "heart_labels.nii.gz"))
    ref = sitk.ReadImage(REFERENCE_LABELS)
    assert new.GetSize() == ref.GetSize()
    assert np.allclose(new.GetSpacing(), ref.GetSpacing()) and np.allclose(new.GetOrigin(), ref.GetOrigin())
    a, b = sitk.GetArrayFromImage(new), sitk.GetArrayFromImage(ref)
    for k, (name, _) in seg.LABELS.items():
        assert dice(a == k, b == k) >= MIN_DICE, f"{name}: Dice {dice(a == k, b == k):.4f}"


@needs_data
def test_procedural_cycle_gives_reference_lv_volumes(pipeline_run):
    info = pipeline_run["info"]
    assert max(info["lv_volume_ml"]) == pytest.approx(REFERENCE_LV_EDV_ML, rel=0.02)
    assert min(info["lv_volume_ml"]) == pytest.approx(REFERENCE_LV_ESV_ML, rel=0.02)
    assert info["lv_ef_percent"] == pytest.approx(REFERENCE_EF, abs=1.0)


@needs_data
def test_all_output_files_written(pipeline_run):
    models, vis = pipeline_run["models"], pipeline_run["vis"]
    expected = ["heart_animated.glb", "heart_animated.gltf", "heart_animated.bin", "heart_viewer.html",
                "heart_labels.nii.gz", "heart_ct_1mm.nii.gz", "segmentation_stats.json", "animation_info.json"]
    expected += [f"heart_{s}.{ext}" for s in seg.SLUGS.values() for ext in ("stl", "obj")]
    missing = [f for f in expected if not (models / f).is_file()]
    missing += [f for f in ("segmentation_overlay.png", "cardiac_cycle.png") if not (vis / f).is_file()]
    assert not missing, f"missing outputs: {missing}"


@needs_data
def test_glb_has_meshes_morph_targets_and_animation(pipeline_run):
    g = read_glb_json(pipeline_run["models"] / "heart_animated.glb")
    assert sorted(m["name"] for m in g["meshes"]) == sorted(name for name, _ in seg.LABELS.values())
    for m in g["meshes"]:
        assert len(m["primitives"][0]["targets"]) == 3, m["name"]
        assert m["extras"]["targetNames"] == ["ventricular_systole", "atrial_systole", "great_artery_distension"]
    anim, = g["animations"]
    assert anim["name"] == "CardiacCycle"
    assert len(anim["channels"]) == len(g["meshes"])
    assert g["extras"]["bpm"] == 72.0


@needs_data
def test_printable_meshes_are_closed_and_match_the_voxels(pipeline_run):
    # STL / OBJ are for 3D printing: closed shells, outward normals, volume close to the label map
    import trimesh
    labels = sitk.GetArrayFromImage(sitk.ReadImage(str(pipeline_run["models"] / "heart_labels.nii.gz")))
    for k, slug in seg.SLUGS.items():
        m = trimesh.load(pipeline_run["models"] / f"heart_{slug}.stl")
        assert m.is_watertight and m.volume > 0, slug
        assert m.volume * 1e6 == pytest.approx((labels == k).sum() / 1000, rel=0.05), slug   # 1 mm voxels


@needs_data
def test_glb_normals_are_unit_length(pipeline_run):
    # glTF requires unit normals (Khronos validator ACCESSOR_VECTOR3_NON_UNIT). Zero-area faces from
    # marching cubes used to leave zero-length ones on the myocardium, which Unity would shade black.
    path = pipeline_run["models"] / "heart_animated.glb"
    g, binary = read_glb_json(path), read_glb_bin(path)
    for m in g["meshes"]:
        a = g["accessors"][m["primitives"][0]["attributes"]["NORMAL"]]
        bv = g["bufferViews"][a["bufferView"]]
        n = np.frombuffer(binary, np.float32, a["count"] * 3, bv["byteOffset"] + a.get("byteOffset", 0)).reshape(-1, 3)
        assert np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-3), m["name"]


# ── unit tests (no dataset needed) ────────────────────────────────────────
def test_lps_to_gltf_axes():
    # LPS millimetres -> glTF metres: +X patient left, +Y superior, +Z anterior
    np.testing.assert_allclose(bhm.lps_to_gltf([1000, 0, 0]), [1, 0, 0])   # left
    np.testing.assert_allclose(bhm.lps_to_gltf([0, 1000, 0]), [0, 0, -1])  # posterior
    np.testing.assert_allclose(bhm.lps_to_gltf([0, 0, 1000]), [0, 1, 0])   # superior


def test_cycle_weights_loop_seamlessly():
    ph, W = bhm.cycle_weights()
    assert ph[0] == 0 and ph[-1] == 1 and W.shape == (len(ph), 3)
    assert W.min() >= 0 and W.max() <= 1
    np.testing.assert_allclose(W[0], W[-1])           # end of beat == start of next
    np.testing.assert_allclose(W[0], 0)               # beat starts at the CT rest pose
    assert ph[W[:, 0].argmax()] < ph[W[:, 1].argmax()]  # ventricles contract before the atrial kick


def test_gltf_builder_writes_valid_glb(tmp_path):
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
    targets = [np.full_like(mesh.vertices, 0.001), np.zeros_like(mesh.vertices)]
    gb = bhm.GLTFBuilder()
    nid = gb.add_mesh("ball", mesh, (1, 0, 0, 1), targets, ["grow", "none"])
    ph, W = bhm.cycle_weights(n=4)
    gb.add_weight_animation("Test", [nid], ph, W[:, :2])
    gb.write_glb(str(tmp_path / "t.glb"))
    g = read_glb_json(tmp_path / "t.glb")
    assert g["meshes"][0]["extras"]["targetNames"] == ["grow", "none"]
    assert g["accessors"][g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["count"] == len(mesh.vertices)
    assert g["animations"][0]["channels"][0]["target"] == {"node": nid, "path": "weights"}
