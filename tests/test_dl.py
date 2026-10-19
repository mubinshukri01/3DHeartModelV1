"""
Tests for the deep-learning label mapping (segment_heart_dl.py).

All synthetic, so they need neither TotalSegmentator nor the dataset: the
aorta is trimmed to its ascending part, the TotalSegmentator classes land
in the ids of segment_heart.LABELS, the SVC of the open task never
overwrites a chamber, and the executable lookup fails with a clear message.
"""
import numpy as np
import pytest

import segment_heart_dl as dl

SP = (1.0, 1.0, 1.0)
CHAMBERS = {"heart_myocardium": 1, "heart_atrium_left": 2, "heart_ventricle_left": 3, "heart_atrium_right": 4,
            "heart_ventricle_right": 5, "aorta": 6, "pulmonary_artery": 7}
TOTAL = {"heart": 51, "aorta": 52, "superior_vena_cava": 62}


def _two_aortas(shape=(40, 40, 40)):
    """Ascending tube (anterior, y 8-13) and descending tube (posterior, y 28-33), plus an LV next to the first."""
    aorta = np.zeros(shape, bool)
    aorta[5:35, 8:14, 10:16] = True
    aorta[0:40, 28:34, 20:26] = True
    lv = np.zeros(shape, bool)
    lv[2:8, 8:14, 17:25] = True          # 1 voxel gap to the ascending tube
    return aorta, lv


def test_ascending_aorta_keeps_the_part_next_to_the_lv():
    aorta, lv = _two_aortas()
    kept = dl.ascending_aorta(aorta, lv, SP)
    assert kept[20, 10, 12] and not kept[20, 30, 22]


def test_ascending_aorta_without_lv_keeps_the_most_anterior_part_and_ignores_specks():
    aorta, _ = _two_aortas()
    aorta[20, 1, 1] = True               # tiny speck, even more anterior
    kept = dl.ascending_aorta(aorta, None, SP)
    assert kept[20, 10, 12] and not kept[20, 30, 22] and not kept[20, 1, 1]


def test_chamber_classes_land_in_the_project_ids_and_svc_never_overwrites_them():
    ch = np.zeros((10, 10, 10), np.uint8)
    for i, v in enumerate(CHAMBERS.values()):
        ch[i, 0:2, 0:2] = v                  # one small block per class, on its own slice
    tot = np.zeros_like(ch)
    tot[3, 0:2, 0:2] = TOTAL["superior_vena_cava"]      # overlaps the right atrium block
    tot[9, 5:7, 5:7] = TOTAL["superior_vena_cava"]      # free space
    labels, sources = dl.project_labels({dl.CHAMBERS_TASK: (ch, CHAMBERS), dl.TOTAL_TASK: (tot, TOTAL)}, SP)
    by_slice = {k: int(labels[k, 0, 0]) for k in range(7)}
    assert by_slice == {0: 4, 1: 2, 2: 1, 3: 5, 4: 6, 5: 3, 6: 7}   # myo, LA, LV, RA, RV, aorta, PA
    assert labels[9, 5, 5] == 8
    assert sources["Aorta"] == f"{dl.CHAMBERS_TASK}:aorta" and sources["Superior vena cava"].startswith("total:")


def test_open_task_alone_gives_only_aorta_and_svc():
    tot = np.zeros((10, 10, 10), np.uint8)
    tot[2:8, 1:3, 1:3] = TOTAL["aorta"]
    tot[2:8, 6:8, 6:8] = TOTAL["superior_vena_cava"]
    tot[0, :, :] = TOTAL["heart"]                        # whole heart is not a project label
    labels, sources = dl.project_labels({dl.TOTAL_TASK: (tot, TOTAL)}, SP)
    assert set(np.unique(labels)) == {0, 3, 8}
    assert set(sources) == {"Aorta", "Superior vena cava"} and sources["Aorta"] == "total:aorta"


def test_find_totalseg_prefers_the_environment_variable_and_explains_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.shutil, "which", lambda name: None)
    monkeypatch.setattr(dl, "DEFAULT_ENV", tmp_path / "no_env")
    monkeypatch.delenv("HEART3D_TOTALSEG", raising=False)
    with pytest.raises(FileNotFoundError, match="HEART3D_TOTALSEG"):
        dl.find_totalseg()
    exe = tmp_path / "TotalSegmentator.exe"
    exe.write_text("")
    monkeypatch.setenv("HEART3D_TOTALSEG", str(exe))
    assert dl.find_totalseg() == exe


def test_find_totalseg_prefers_the_project_environment_over_path(tmp_path, monkeypatch):
    other = tmp_path / "Python310" / "Scripts" / "TotalSegmentator.exe"      # e.g. a CPU-only copy on PATH
    other.parent.mkdir(parents=True)
    other.write_text("")
    (tmp_path / "Python310" / "python.exe").write_text("")
    monkeypatch.setattr(dl.shutil, "which", lambda name: str(other))
    monkeypatch.delenv("HEART3D_TOTALSEG", raising=False)
    env = tmp_path / "heart3d-dl"
    monkeypatch.setattr(dl, "DEFAULT_ENV", env)
    assert dl.find_totalseg() == other                                        # no project environment: PATH
    assert dl.env_python(other) == tmp_path / "Python310" / "python.exe"      # system layout: python one folder up
    own = env / ("Scripts" if dl.os.name == "nt" else "bin") / other.name
    own.parent.mkdir(parents=True)
    own.write_text("")
    (own.parent / ("python.exe" if dl.os.name == "nt" else "python")).write_text("")
    assert dl.find_totalseg() == own                                          # the project's GPU environment first
    assert dl.env_python(own) == own.with_name("python.exe" if dl.os.name == "nt" else "python")


def test_dice():
    a = np.zeros(10, bool); a[:4] = True
    b = np.zeros(10, bool); b[2:6] = True
    assert dl.dice(a, b) == pytest.approx(0.5) and dl.dice(a & False, b & False) is None
