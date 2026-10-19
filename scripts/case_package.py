"""
==============================================================
  STEP 2b - Case package for Unity / XR (CardioTwin XR roadmap, step 11)
  UTM-Claude Research Project | rohayanti@utm.my

  Bundles one case for the XR app: the animated model, a case card and
  the CT on the model's grid, so the app's Case panel and slice tool need
  nothing else. No patient identifiers go in: the case has its own ID
  (--case-id), and the card holds no names, IDs or scan dates.

  Usage (after build_heart_model.py):
    python case_package.py [--case-id DEMO-001] [--no-light]
  Output (output/case_package/<case id>/):
    heart.glb    the animated model (heart_animated.glb)
    heart_light.glb  the same model with about a quarter of the triangles
                 (marching cubes at 2 voxels), for standalone headsets
    case.json    structures (volume, colour, provenance, an inside point),
                 measurements (EAT, calcium score), simulated motion, notice,
                 and the volume grid with the model -> voxel transforms
    ct.raw       the 1 mm CCTA the model was built from: uint16 little-endian,
                 HU + 1024, x fastest, then y, then z (Unity Texture3D order)
    labels.raw   the model's labels on the same grid, uint8, same order
==============================================================
"""
import argparse, datetime, json, os, shutil, struct, tempfile

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

import build_heart_model as bhm
import calcium_score as cs
import heart_io as hio
import vessels as vs

SCHEMA = "cardiotwin-case/1"
DEFAULT_CASE_ID = "DEMO-001"
HU_OFFSET = 1024            # stored value = HU + 1024 (uint16)
NOTICE = "Research prototype. Simulated motion. Not for clinical use."
WINDOWS = [("CCTA", 300, 800), ("Soft tissue", 40, 400), ("Fat", -100, 300)]     # name, centre, width (HU)
GLTF_TO_LPS = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])   # glTF axes -> LPS axes (see bhm.gltf_to_lps)
GLTFAST_MIRROR = np.diag([-1.0, 1, 1])                          # Unity (glTFast) position -> glTF position
LIGHT_FACES = 120000        # face budget of the light model: build_heart_model then meshes at 2 voxels
MODEL_INPUTS = ["heart_labels.nii.gz", "heart_ct_1mm.nii.gz", "segmentation_stats.json", "vessels_dl.nii.gz",
                "vessels_dl.json", "eat_labels.nii.gz", "eat_stats.json", "heart_labels_dl.nii.gz", "dl_segmentation.json",
                "coronaries_dl.nii.gz", "coronaries_dl.json"]


def out_root():
    return os.path.join(os.path.dirname(hio.OUT_MODELS), "case_package")


def read_glb_json(path):
    with open(path, "rb") as fh:
        magic, version, _ = struct.unpack("<III", fh.read(12))
        if magic != 0x46546C67 or version != 2:
            raise ValueError(f"{path} is not a glTF 2.0 binary")
        n, _ = struct.unpack("<II", fh.read(8))
        return json.loads(fh.read(n))


def voxel_from_model(img, origin_lps_mm, unity=False):
    """4x4 (row-major) mapping model positions (m; glTF, or Unity after glTFast's X mirror) to continuous voxel
    indices (x, y, z) of img: LPS = origin_lps + 1000 * GLTF_TO_LPS @ model, then LPS -> index."""
    D = np.array(img.GetDirection()).reshape(3, 3)
    to_index = np.diag(1.0 / np.array(img.GetSpacing())) @ D.T
    A = to_index @ (1000.0 * GLTF_TO_LPS)
    if unity:
        A = A @ GLTFAST_MIRROR
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = A, to_index @ (np.asarray(origin_lps_mm, float) - np.array(img.GetOrigin()))
    return M


def model_from_index(img, origin_lps_mm, idx_zyx):
    """glTF model positions (m) of voxel indices given as (z, y, x) rows."""
    lps = np.array([img.TransformContinuousIndexToPhysicalPoint(i[::-1].tolist()) for i in np.asarray(idx_zyx, float)])
    return bhm.lps_to_gltf(lps) - bhm.lps_to_gltf(np.asarray(origin_lps_mm, float))


def triangles(gltf_json):
    return sum(gltf_json["accessors"][p["indices"]]["count"] // 3 for m in gltf_json["meshes"] for p in m["primitives"])


def build_light(dest, bpm, chambers, log=print):
    """The model with the light face budget, built from the same inputs (and chamber labels) in a temporary folder,
    saved as dest."""
    saved = hio.OUT_MODELS, hio.OUT_VIS
    with tempfile.TemporaryDirectory(prefix="heart_light_") as tmp:
        for name in MODEL_INPUTS:
            if os.path.exists(os.path.join(hio.OUT_MODELS, name)):
                shutil.copy(os.path.join(hio.OUT_MODELS, name), tmp)
        try:
            hio.OUT_MODELS = hio.OUT_VIS = tmp
            bhm.main(bpm=bpm, total_faces=LIGHT_FACES, log=lambda *a, **k: None, chambers=chambers)
        finally:
            hio.OUT_MODELS, hio.OUT_VIS = saved
        shutil.copyfile(os.path.join(tmp, "heart_animated.glb"), dest)


def write_raw(path, arr, dtype):
    """(z, y, x) array -> raw little-endian file, x fastest."""
    np.ascontiguousarray(arr, dtype=np.dtype(dtype).newbyteorder("<")).tofile(path)


def provenance(k, has_vessels, deep=False):
    """(kind, method) of model structure k: how its shape was obtained (deep: the deep-learning chambers)."""
    if k == bhm.EAT_LABEL:
        return ("display layer", "CCTA fat that touches the heart near the epicardial fat measured on the calcium-score "
                "scan (fat_layer). For viewing; the measured volume is under measurements.")
    if k == bhm.COR_LABEL:
        return ("segmented", "Deep learning: TotalSegmentator coronary_arteries (coronaries.py), predicted at 0.7 mm and meshed "
                "at that detail, finer than the 1 mm heart. Main branches; small distal branches may be missing or broken up.")
    if k in (9, 10, 11) or (k == 8 and (has_vessels or deep)):
        return "segmented", "TotalSegmentator, open 'total' task (vessels.py). Ends where the scan ends."
    if deep:
        method = "Deep learning: TotalSegmentator heartchambers_highres, sub-millimetre (segment_heart_dl.py)"
        if k == 4:
            method += "; the LV wall"
        if k == 3:
            method += "; the ascending aorta and root"
        if has_vessels and k == 2:
            method += "; left atrial appendage from the 'total' task"
        return "segmented", method + "."
    if k == 8:
        return "approximate", "Rule-based estimate (segment_heart.py); lies partly in the right atrial appendage."
    method = "Rule-based segmentation of the CCTA (segment_heart.py)"
    if k == 4:
        method += "; the label also covers thin sheets around the LA and the aortic root"
    if has_vessels and k in (2, 3):
        method += {2: "; left atrial appendage from TotalSegmentator", 3: "; aortic root from TotalSegmentator"}[k]
    return "segmented", method + "."


def _load(name):
    path = os.path.join(hio.OUT_MODELS, name) if not os.path.isabs(name) else name
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def measurements():
    """Case-panel entries: name, value (if any), unit, display text, status, source."""
    out = []
    eat = _load("eat_stats.json")
    if eat:
        src = (f"{eat['series']}, {eat['cardiac_phase_percent']:g} % R-R; EAT model trained on "
               f"{eat['model']['trained_on_patients']} patients (benchmark Dice {eat['model']['benchmark_dice_mean']:g}). "
               "Research estimate.")
        out.append(dict(name="Epicardial fat volume", value=eat["eat_volume_ml"], unit="mL",
                        text=f"{eat['eat_volume_ml']:g} mL", status="measured", source=src))
        out.append(dict(name="Epicardial fat attenuation", value=eat["eat_mean_hu"], unit="HU",
                        text=f"{eat['eat_mean_hu']:g} HU", status="measured", source=src))
    ca = _load("calcium_stats.json")
    if ca:
        auto = ca["automatic"]
        src = f"{ca['series']}, threshold {ca['criteria']['threshold_hu']} HU (calcium_score.py)"
        out.append(dict(name="Calcium score (automatic)", value=auto["agatston"], unit="Agatston",
                        text=f"{auto['agatston']:g} · CAC-DRS {auto['cac_drs']}", status="automatic, not reviewed",
                        source=src))
        review = _load(cs.review_path())
        if review is not None:
            a, _, n = cs.score(ca["lesions"], set(review["coronary"]))
            out.append(dict(name="Calcium score (reviewed)", value=a, unit="Agatston",
                            text=f"{a:g} · CAC-DRS {cs.cac_drs(a)}", status="reviewed",
                            source=f"{n} lesion(s) confirmed by a reader"))
        if ca.get("report_agatston") is not None:
            out.append(dict(name="Calcium score (radiology report)", value=ca["report_agatston"], unit="Agatston",
                            text=f"{ca['report_agatston']:g}", status="reported", source="radiology report"))
    out.append(dict(name="CAD model", unit="", text="In development", status="in development", source=""))
    return out


def main(case_id=DEFAULT_CASE_ID, out_dir=None, light=True, log=print):
    glb = os.path.join(hio.OUT_MODELS, "heart_animated.glb")
    if not os.path.exists(glb):
        raise FileNotFoundError("run build_heart_model.py first (needs heart_animated.glb)")
    g = read_glb_json(glb)
    ex = g.get("extras", {})
    if "origin_lps_mm" not in ex:
        raise ValueError("heart_animated.glb predates the case package (no origin_lps_mm): re-run build_heart_model.py")
    origin = ex["origin_lps_mm"]
    folder = os.path.join(out_dir or out_root(), case_id)
    os.makedirs(folder, exist_ok=True)
    log(f"[1/3] Labels and CT for case {case_id} ...")
    lab_img = sitk.ReadImage(os.path.join(hio.OUT_MODELS, "heart_labels.nii.gz"))
    source = "dl" if ex.get("chambers", "").startswith("deep") else "rules"     # the labels the model was built from
    labels, ves, _ = bhm.model_labels(lab_img, log=lambda *_: None, chambers=source)
    vox_ml = float(np.prod(lab_img.GetSpacing())) / 1000.0
    for name, ml in ex.get("model_volumes_ml", {}).items():      # the labels must be the ones the model was built from
        k = next(k for k, v in bhm.MODEL_LABELS.items() if v[0] == name)
        if abs((labels == k).sum() * vox_ml - ml) > 0.051:
            raise ValueError(f"{name}: the labels differ from the model's ({ml} mL): re-run build_heart_model.py")
    ct_img = sitk.ReadImage(os.path.join(hio.OUT_MODELS, "heart_ct_1mm.nii.gz"))
    if ct_img.GetSize() != lab_img.GetSize() or not np.allclose(ct_img.GetOrigin(), lab_img.GetOrigin()):
        raise ValueError("heart_ct_1mm.nii.gz and heart_labels.nii.gz are on different grids: re-run segment_heart.py")
    hu = np.rint(sitk.GetArrayFromImage(ct_img)).clip(-HU_OFFSET, 65535 - HU_OFFSET) + HU_OFFSET
    write_raw(os.path.join(folder, "ct.raw"), hu, np.uint16)
    write_raw(os.path.join(folder, "labels.raw"), labels, np.uint8)

    log("[2/3] Case card ...")
    names = [m["name"] for m in g["meshes"]]
    structures = []
    for name in names:
        k = next(k for k, v in bhm.MODEL_LABELS.items() if v[0] == name)
        kind, method = provenance(k, ves is not None, source == "dl")
        inside = ndi.distance_transform_edt(np.pad(labels == k, 1), sampling=lab_img.GetSpacing()[::-1])[1:-1, 1:-1, 1:-1]
        anchor = model_from_index(lab_img, origin, [np.unravel_index(np.argmax(inside), labels.shape)])[0]
        structures.append(dict(name=name, label=k, slug=bhm.MODEL_SLUGS[k], rgba=list(bhm.MODEL_LABELS[k][1]),
                               volume_ml=ex["model_volumes_ml"][name], kind=kind, method=method,
                               anchor_m=[round(float(v), 5) for v in anchor]))
    info = _load("animation_info.json") or {}
    lv = info.get("lv_volume_ml", [])
    flat = lambda M: [round(float(v), 9) for v in M.ravel()]
    case = {
        "schema": SCHEMA, "case_id": case_id, "created": datetime.date.today().isoformat(), "notice": NOTICE,
        "model": {"file": "heart.glb", "units": "m", "triangles": triangles(g),
                  "axes": "glTF: +X patient left, +Y superior, +Z anterior. Unity (glTFast import): +X patient right.",
                  "origin_lps_mm": origin, "series": ex.get("source_series", ""), "rest_phase": ex.get("rest_phase", "")},
        "motion": {"kind": "simulated (procedural morph targets)", "bpm": ex.get("bpm", 0), "period_s": ex.get("period_s", 0),
                   "clip": "CardiacCycle", "targets": list(bhm.TARGETS),
                   "lv_edv_ml": round(max(lv), 1) if lv else 0, "lv_esv_ml": round(min(lv), 1) if lv else 0,
                   "lv_ef_percent": round(info.get("lv_ef_percent", 0), 1)},
        "structures": structures,
        "measurements": measurements(),
        "grid": {"dims": list(lab_img.GetSize()), "spacing_mm": list(lab_img.GetSpacing()),
                 "origin_lps_mm": list(lab_img.GetOrigin()), "direction": list(lab_img.GetDirection()),
                 "order": "x fastest, then y, then z",
                 "voxel_from_gltf": flat(voxel_from_model(lab_img, origin)),
                 "voxel_from_unity": flat(voxel_from_model(lab_img, origin, unity=True))},
        "ct": {"file": "ct.raw", "dtype": "uint16", "byte_order": "little", "hu_offset": -HU_OFFSET,
               "series": ex.get("source_series", ""),
               "windows": [dict(name=n, center=c, width=w) for n, c, w in WINDOWS]},
        "labels": {"file": "labels.raw", "dtype": "uint8", "note": "values are the structures' label numbers; 0 = none"},
    }
    with open(os.path.join(folder, "case.json"), "w") as fh:
        json.dump(case, fh, indent=2)
    log("[3/3] Model" + (" and the light model (about 15 s) ..." if light else " ..."))
    shutil.copyfile(glb, os.path.join(folder, "heart.glb"))
    files = ["heart.glb", "ct.raw", "labels.raw", "case.json"]
    light_path = os.path.join(folder, "heart_light.glb")
    if light:
        build_light(light_path, ex.get("bpm", 72.0), source, log)
        gl = read_glb_json(light_path)
        if gl.get("extras", {}).get("origin_lps_mm") != origin:
            raise ValueError("the light model is centred differently from the model")
        case["model"]["light_file"] = "heart_light.glb"
        case["model"]["light_triangles"] = triangles(gl)
        with open(os.path.join(folder, "case.json"), "w") as fh:
            json.dump(case, fh, indent=2)
        files.append("heart_light.glb")
    elif os.path.exists(light_path):
        os.remove(light_path)                                          # a stale light model would not match
    mb = sum(os.path.getsize(os.path.join(folder, f)) for f in files) / 1e6
    log(f"Done -> {folder} ({mb:.1f} MB): {len(structures)} structures, {len(case['measurements'])} measurements, "
        f"CT {case['grid']['dims']}")
    return case


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", default=DEFAULT_CASE_ID, help="anonymous case ID shown in the app")
    ap.add_argument("--out", help="output folder (default: output/case_package)")
    ap.add_argument("--no-light", action="store_true", help="leave out the light model for standalone headsets")
    a = ap.parse_args()
    main(a.case_id, a.out, light=not a.no_light)
