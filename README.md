# 3D Heart Model V1

UTM-Claude Research Project · Assoc Prof Ts Dr Rohayanti Hassan (rohayanti@utm.my)

Builds a 3D, beating model of a patient's heart from a cardiac CT scan, measures the epicardial fat and the coronary calcium score, and shows everything in a web dashboard (Streamlit) and a Unity app ("CardioTwin XR").

This page shows how to run the project. The methods, results and design decisions are in [README_3D.md](README_3D.md).

## What you need

| For | You need |
|---|---|
| The pipeline, tests and dashboard | Windows, Python 3.9, and the CT data (below) |
| The deep-learning steps (optional) | An NVIDIA GPU (6 GB is enough), Python 3.12, TotalSegmentator, and your own free TotalSegmentator licence |
| The Unity app (optional) | Unity 6000.6.3f1 |

All commands below are typed in PowerShell, in the project folder (the one with this file).

## 1. Get the code and set up Python

```
git clone https://github.com/mubinshukri01/3DHeartModelV1.git
cd 3DHeartModelV1
```
Clone it into a short path such as `C:\code\`: Windows' 260-character path limit can break Python packages installed in very deep folders.

With [uv](https://docs.astral.sh/uv/) (recommended):
```
uv venv .venv --python 3.9
uv pip install --python .venv\Scripts\python.exe -r requirements_dev.txt
```
Without uv, with Python 3.9 installed:
```
py -3.9 -m venv .venv
.venv\Scripts\pip install -r requirements_dev.txt
```
`requirements_dev.txt` installs the exact tested versions (`requirements_3d.lock.txt`) plus pytest.

## 2. Add the data (not in this repository)

The patient CT data is **not on GitHub** and must never be committed. Get a copy from the project lead and put it in `dataset/`, with these folder names:

```
dataset/
  CT CCTA DL/          contrast CT (DICOM): the 3D heart is built from this one   (required)
  CT CALCIUM SCORE/    non-contrast CT: epicardial fat and calcium score          (optional)
  Rodriguez/           fat benchmark dataset: only to retrain the fat model       (optional)
```
The other series (lungs, mediastinum, scout, reports) are only for viewing in the dashboard. `dataset/` and `output/` are listed in `.gitignore`, so git leaves them out.

## 3. Run the tests

```
.venv\Scripts\python -m pytest
```
Without the data, 75 tests pass and 14 are skipped (they need the CT). With the data and a pipeline run (step 4), all 89 run. About 2.5 minutes.

## 4. Run the pipeline

Each step reads the CT from `dataset/` and the previous steps' results from `output/3d_models/`. Run them in this order:

```
.venv\Scripts\python scripts\segment_heart.py       # 1  outline the heart structures (~10 s)
.venv\Scripts\python scripts\eat_patient.py         # 1b epicardial fat     (optional, needs CT CALCIUM SCORE)
.venv\Scripts\python scripts\calcium_score.py       # 1c Agatston score     (optional, after 1b)
.venv\Scripts\python scripts\build_heart_model.py   # 2  3D meshes + heartbeat -> heart_animated.glb (~15 s)
.venv\Scripts\python scripts\case_package.py        # 2b bundle for the Unity app
```

Results:
- `output/3d_models/heart_animated.glb`: the animated 3D heart (also `.stl`/`.obj` per structure for 3D printing).
- `output/case_package/DEMO-001/`: what the Unity app loads.
- `output/visualizations/`: quality-check figures. Look at `segmentation_overlay.png` after every new scan.

### Optional: the deep-learning steps (better chambers, vessels, coronary arteries)

TotalSegmentator needs Python 3.12 and PyTorch, so it gets its own environment, outside the project (about 5 GB):
```
uv venv $env:USERPROFILE\venvs\heart3d-dl --python 3.12
uv pip install --python $env:USERPROFILE\venvs\heart3d-dl\Scripts\python.exe --torch-backend auto torch TotalSegmentator
```
The chambers and coronary arteries need a TotalSegmentator licence, free for non-commercial use. Get **your own** at https://backend.totalsegmentator.com/license-academic/ and set it once (never commit it):
```
& "$env:USERPROFILE\venvs\heart3d-dl\Scripts\totalseg_set_license.exe" -l <your-key>
```
Then, after `segment_heart.py` and before `build_heart_model.py`:
```
.venv\Scripts\python scripts\segment_heart_dl.py    # deep-learning chambers (~7 min on an RTX 2060)
.venv\Scripts\python scripts\vessels.py             # great vessels (~75 s; no licence needed)
.venv\Scripts\python scripts\coronaries.py          # coronary arteries (~3.5 min)
```
`build_heart_model.py` uses their results automatically when they exist (`--chambers rules` forces the rule-based chambers).

## 5. Open the dashboard

```
.venv\Scripts\streamlit run scripts\heart_dicom_viewer.py
```
Then open http://localhost:8501.
- **🪟 DICOM Multi-View:** enter the path of your `dataset` folder and pick a CT series.
- **🫀 3D Heart Model:** the beating 3D heart, volumes, fat and the calcium review. **▶ Run pipeline** (shown when `dataset/` has CT series) runs steps 1, 1b, 1c, 2 and 2b for you; it reuses the deep-learning results if you made them.

## 6. Open the Unity app (optional)

1. Create a Unity **6000.6.3f1** project (built-in render pipeline; Linear colour space recommended).
2. In **Window → Package Manager → + → Add package by name**, add `com.unity.cloud.gltfast` version `6.20.0` and `com.unity.ugui` version `2.6.0`.
3. Copy `unity/Assets/Heart/` from this repository into the Unity project's `Assets/`.
4. Copy `output/3d_models/heart_animated.glb` into `Assets/Heart/`, and `output/case_package/DEMO-001/` into `Assets/StreamingAssets/Cases/DEMO-001/`. Copy them again after every pipeline run.
5. In Unity: **Tools → Heart → Run import check** (all checks should pass), then **Tools → Heart → Create demo scene**.
6. Open `Assets/Heart/HeartDemo.unity` and press **Play**. Drag to rotate, scroll to zoom, right-drag to move, R to reset; the panels show the layers, the case and the slice tool.

## What's where

| Folder | Contents |
|---|---|
| `scripts/` | The pipeline (`segment_heart.py` … `case_package.py`), the dashboard (`heart_dicom_viewer.py`, `heart_3d_tab.py`, `slice_view.py`, `overlay.py`) and the fat-model research (`eat_segment.py`, `eat_benchmark.py`, `rodriguez.py`). `heart_dicom_viewer_v2.py` and `heart_dicom_viewer_backup.py` are old versions, not used. |
| `tests/` | The automatic checks (pytest). `tests/reference/` holds the reference result the pipeline test compares against. |
| `models/` | `eat_classifier.joblib`: the trained epicardial-fat model (retrain with `scripts\eat_benchmark.py --save-model`, ~50 min, needs `dataset/Rodriguez`). |
| `unity/Assets/Heart/` | The Unity app: `Scripts/`, `Editor/` (the checks under Tools → Heart) and `Shaders/`. |
| `dataset/`, `output/` | Your local data and results. Not in the repository. |

## More
- [README_3D.md](README_3D.md): how each step works, the results for the test patient, limitations, and the Unity details.
- `requirements_3d.txt`: the direct dependencies. After changing it, regenerate the lock file (see README_3D.md, "Dependencies") and re-run the tests.
