# Corrections for the PDF manuals

The two PDF manuals (`3DHeartApp_UserManual.pdf`, `3DHeartApp_AdminManual.pdf`) were exported from Microsoft Word. The `.docx` sources are not in this project, so the PDFs themselves have not been changed. Apply these corrections to the Word files, then export again. Every "Correct" entry was checked against the code, the data or a real run on 26 Sep 2026.

`README.txt` and `README_3D.md` are already up to date and can be used as the source for the new install, test and tuning text.

---

## User Manual (`3DHeartApp_UserManual.pdf`)

| Page · Section | Manual says | Correct |
|---|---|---|
| p.2 · §1 steps 2–3 | `cd C:\3DHeartModelV1\scripts`, then `streamlit run heart_dicom_viewer.py` | The project can be in any folder. From the project folder: `.venv\Scripts\streamlit run scripts\heart_dicom_viewer.py`. One-time Python setup is in `README.txt`. |
| p.2 · tab table | Tab "🖼️ DICOM Viewer" | The tab is called **🪟 DICOM Multi-View** |
| p.3 · §2.1 Path Folder | Example `C:\3DHeartModelV1\dataset\CT_CCTA_DL` | Folder names contain **spaces**: `…\dataset\CT CCTA DL`. Easiest: enter the `dataset` folder itself and pick the series from the dropdown. The box's grey placeholder shows the correct path. |
| p.5 · §3.1 Reset View | Double-click the background | Click the **Reset view** button in the viewer's panel. Double-click does nothing. |
| p.5–6 · §3.2 colours | LA = Pink; Aorta = Orange / Bright Red | LV = red · **LA = blue** · **Aorta = golden yellow** · Myocardium & right heart = semi-transparent dark red |
| p.6 · §3.3 | "Click the Play button" | The animation **starts automatically**. The button switches between Pause and Play. |
| p.6 · §3.3 phase table | Diastole 0–40 %, Atrial systole 40–50 %, Systole 50–75 %, Isovolumetric relaxation 75–100 % | Phase 0 % = ECG R wave. 0–5 % isovolumic contraction · **5–33 % ejection (systole)** · 33–42 % isovolumic relaxation · 42–60 % rapid filling · 60–80 % diastasis (the CT rest pose, 75 %) · **80–98 % atrial systole** · 98–100 % isovolumic contraction. The viewer shows the current phase name under the controls. |
| p.6 · Note | Default heart rate 60 bpm | Default **72 bpm**. The slider goes from 30 to 150 bpm. |
| p.6–7 · §3.4 | EDV / ESV / EF are shown below the animation controls | The side panel shows the **segmented volumes** (LV 33.0, LA 61.1, Aorta 23.7, Myocardium + right heart 207.6, whole heart 325.4 mL) and one line with bpm, period, model EF (≈ 60 %) and GLB size. EDV (35.3 mL) and ESV (14.3 mL) are the top and bottom of the LV volume curve in the "📈 Cardiac cycle" expander, and are stored in `output\3d_models\animation_info.json`. |
| p.7 · §3.5 Downloads | Lists heart_la.stl, heart_aorta.stl and heart_viewer.html | The app offers **heart_animated.glb, heart_lv.stl, heart_myocardium.stl**. The other files (including .obj versions) are in `output\3d_models\`. |
| p.7 · §3.5 heart_viewer.html | "Open in any browser — no internet connection or app needed" | It must be **served over http**: run `python -m http.server` in `output\3d_models`, then open `http://localhost:8000/heart_viewer.html`. It **needs internet** to load three.js. Double-clicking the file shows "Could not load…". |
| p.7 · §3.6 | Button "Run Full Pipeline"; takes 30–60 s | Button **"▶ Run pipeline"**. About 10 s on a typical PC, 35–45 s on an ARM laptop. It only appears when the `dataset` folder is present, so it is hidden on the web server. |
| p.9 · FAQ "3D tab is blank" | "▶ Run Full Pipeline", wait about 60 s | "▶ Run pipeline", about 10–45 s |
| p.9 · FAQ "black screen" | `pip install SimpleITK`; sidebar shows "SimpleITK ✅" | Install from `requirements_3d.lock.txt` (it includes SimpleITK) and restart. The sidebar then shows **"SimpleITK ✅ — compressed DICOM supported"**. |
| p.10 · FAQ "typed the path" | `…\CT_CCTA_DL\`; dataset must be in `C:\3DHeartModelV1\dataset\` | `…\CT CCTA DL\` (spaces), or just the `dataset` folder. The project can be anywhere. |

---

## Administrator & Technical Manual (`3DHeartApp_AdminManual.pdf`)

| Page · Section | Manual says | Correct |
|---|---|---|
| p.1 | Version 1.0 · September 2026 | Raise to **1.1** once these corrections are applied |
| p.2 · §1.1 Hardware | NVIDIA GPU "required for fast segmentation", 6–8 GB VRAM, 16 GB RAM minimum | The V1 pipeline is **CPU-only**: no PyTorch, no GPU. Measured peak memory is **0.64 GB**, so any machine with 4 GB RAM or more is enough. A GPU is only needed later, for the planned deep-learning milestones (TotalSegmentator / nnU-Net). |
| p.2 · §1.2 Software | CUDA Toolkit, Git for model weights; Streamlit 1.32+ | CUDA and Git are not needed for V1. Tested with **Python 3.9.25 and Streamlit 1.50.0**. All exact versions are in `requirements_3d.lock.txt`. `uv` is recommended for setup. |
| p.3 · §2.1 | Verify GPU & CUDA | Not needed for V1. Remove this step or mark it "future DL milestones only". |
| p.3 · §2.2 | `pip install … torch …`, `TotalSegmentator pyvista vtk open3d` | `uv pip install --python .venv\Scripts\python.exe -r requirements_3d.lock.txt` (or `pip install -r requirements_3d.lock.txt`). torch, TotalSegmentator, pyvista, vtk and open3d are **not used** by the code. |
| p.3 · §2.3 | "All CT series use **JPEG2000** compression"; path `…\\CT_CCTA_DL\\CT000000.dcm` | The files use **JPEG Lossless, Non-Hierarchical, First-Order Prediction** (transfer syntax 1.2.840.10008.1.2.4.70). JPEG2000 would be …4.90. SimpleITK decodes it. Path: `dataset\CT CCTA DL\CT000000.dcm`. The expected output `(512, 512, 1)` is correct. |
| p.4 · §2.4 folder tree | `C:\3DHeartModelV1\`, `CT_CCTA_DL`, `CT_CALCIUM_SCORE` (100 slices), `CT_SSFreeze_116` (300 slices) | The project can be anywhere. The names contain spaces: `CT CCTA DL` (224), `CT CALCIUM SCORE` (**50**), `CT SSFreeze 116 Original Series 302` (**224**). Also add `scripts\heart_io.py`, `tests\`, `deploy\` and `dataset\Rodriguez\`. |
| p.5 · §3.1 "Threshold Used" column | >350 HU contrast pool · 80–350 HU envelope · liver removed by z-position + HU · lung/fat excluded < 0 HU · spine > 400 HU · calcium > 600 HU → separate CAC mask | After smoothing: contrast pool **300–700 HU** (`CONTRAST_LOWER/UPPER`). Heart envelope **110–700 HU** (`ENVELOPE_LOWER/UPPER`), which also excludes fat and lung. Liver and chest wall are separated by **erosion** (`EROSION_RADIUS` = 4) and by keeping only the part connected to the blood pool, with **no z-position rule**. Bone = regions enclosed by > 600 HU, plus anything > 900 HU (`BONE_FILL_HU/BONE_HU`). There is **no CAC mask**; calcium beside the blood pool is kept out of the bone mask. (The "Measured HU Range" column was not re-checked.) |
| p.5 · §3.2 steps 1–5 | Check HU by hovering; edit `CONTRAST_LOWER` / `ENVELOPE_LOWER` in "lines 45–52"; `EROSION_ITER` default 2 | The viewer has **no hover readout**. It shows each slice's min–max HU. For point values, open `output\3d_models\heart_ct_1mm.nii.gz` in 3D Slicer or ITK-SNAP. The constants are at the **top of `segment_heart.py` (lines 40–51)**. `EROSION_RADIUS` (default **4**) replaces `EROSION_ITER`. After a change, run `python -m pytest`; it fails on purpose, and if the change is intended you update the test reference (see README_3D.md, "Tests"). |
| p.5–6 · code block | `CONTRAST_LOWER = 350`, `CONTRAST_UPPER = 1200`, `ENVELOPE_LOWER = 80`, `ENVELOPE_UPPER = 350`, `EROSION_ITER = 2`, `LIVER_Z_FRAC = 0.35`, `SPINE_X_FRAC = 0.70` | Replace with the real block from `segment_heart.py` lines 40–51. `LIVER_Z_FRAC` and `SPINE_X_FRAC` **do not exist**. |
| p.6 · §3.3 | Presets "can be changed per-window in the app sidebar" | Each window has **its own** preset dropdown. The sidebar's "Sync Preset" applies one preset to all windows. |
| p.7 · §4.1 | `cd C:\3DHeartModelV1\scripts` … | Same commands as `README.txt` / `README_3D.md` (run from the project folder with `.venv\Scripts\…`) |
| p.7 · §4.2 runtimes | segment 30–60 s (GPU) / 5–10 min (CPU); build 15–30 s | CPU only. Segment **~8 s** (x86 PC) or **35–45 s** (ARM laptop under emulation). Build **2–7 s**. Test suite about 1.5 min. |
| p.7 · tip | "▶ Run Full Pipeline" | "▶ Run pipeline" |
| p.8 · §5.3 | 3D viewer uses "WebGL via **Plotly**/Three.js"; "not designed for multi-user deployment without a dedicated server" | The main app uses **three.js only**; Plotly appears only in the experimental `heart_dicom_viewer_v2.py`. A server deployment **exists** in `deploy\`: nginx reverse proxy with a password prompt, plus a systemd service. HTTPS is not enabled yet. |
| p.8–9 · §6.1 | "JPEG2000 lossless (…1.2.4.70)"; use `download_smart.py` | …1.2.4.70 is **JPEG Lossless (Process 14)**, not JPEG2000. `download_smart.py` is **not in the project**; use `scripts\download_dataset.py` or `scripts\run_everything.py`. |
| p.9 · §6.2 | "Liver exclusion by z-position is fragile…" | Keep as a lesson, but mark it historical: the current code uses erosion and connectivity, not z-position. |
| new sections | — | Add **Dependencies** (lock file), **Tests** (pytest regression suite) and **Tuning the segmentation** (constants table). Copy them from `README_3D.md`. |
