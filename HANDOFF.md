# Handoff: continuing on the office PC (NVIDIA GPU)

Written 26 Sep 2026 at the end of a Claude Code session on the ARM laptop. It lets the next session, on the office PC, carry on without the earlier conversation. **Claude: read this file, then `README_3D.md`, before doing anything else.**

## 1. Copying the project
- **Copy:** the whole `3DHeartModelV1` folder (about 1.4 GB, almost all of it `dataset/`).
- **Leave behind:**
  - `.venv/`: rebuild it on the office PC (section 2).
  - `__pycache__/` and `.pytest_cache/`.
  - `Jangan delete.txt`: it holds the VPS root password in plain text. Keep it off extra machines, and preferably move the password to a password manager.
- **Consider first:** the privacy redaction (section 5).

## 2. Setup on the office PC
Python **3.9** matches the lock file exactly. With [uv](https://docs.astral.sh/uv/), from the `3DHeartModelV1` folder:
```
uv venv .venv --python 3.9
uv pip install --python .venv\Scripts\python.exe -r requirements_dev.txt
.venv\Scripts\python -m pytest            # expect 30 passed, ~3 min
.venv\Scripts\streamlit run scripts\heart_dicom_viewer.py
```
- **Checks.** All 30 tests should pass. The app should show the 3D heart with the epicardial fat, and the coronary-calcium panel.
- **If a test fails,** stop and find out why before changing anything.
- **Launch config.** On the laptop the app was started from a Claude Code launch config in the folder *above* `3DHeartModelV1` (`.claude/launch.json`: name `heart-app`, command `3DHeartModelV1\.venv\Scripts\streamlit.exe run 3DHeartModelV1\scripts\heart_dicom_viewer.py --server.headless true --server.port 8501`). Recreate it if it's wanted.

## 3. Where things stand
The user was onboarded on 26 Sep 2026 and agreed a 16-item improvement list. **Work one item at a time, and only after the user says go.**

| # | Item | Status |
|---|---|---|
| 1 | Put the project under git | done 19 Oct 2026: private repo https://github.com/mubinshukri01/3DHeartModelV1 (branch `main`). `.gitignore` keeps out `dataset/`, `output/`, `Jangan delete.txt`, `deploy/` (VPS address), the manuals, `.venv/` and stray Unity caches |
| 2 | Secure the VPS (plain-text root password, HTTP only) | not requested yet |
| 3 | pytest regression suite (`tests/`) | done |
| 4 | Named HU thresholds in `segment_heart.py` | done |
| 5 | Pinned versions (`requirements_3d.lock.txt`), no hard-coded paths | done |
| 6 | READMEs fixed; `3DHeartApp_Manual_Corrections.md` for the Word/PDF manuals | done (the `.docx` sources of the manuals are not in the project) |
| 7 | EAT segmentation, validated on the Rodriguez dataset | done: Dice 0.805 ± 0.055 (leave-one-patient-out, 18 patients); rules only 0.652 |
| 8 | EAT for TC_PTID, shown in the 3D model | done: 73.4 mL, −78.5 HU (from CT CALCIUM SCORE) |
| 9 | Agatston calcium score | done: semi-automatic. The automatic score is 17.5 (5 faint edge specks); the report is 0; review gives 0 |
| **10** | **Deep-learning heart-chamber segmentation (TotalSegmentator / nnU-Net)** | **in progress on the office PC (29 Sep): `scripts/segment_heart_dl.py` + `tests/test_dl.py` done and run with the open task (aorta Dice 0.88; the rule-based SVC looks misplaced). Waiting for the user's `heartchambers_highres` licence to run the chambers. Details: README_3D.md, "Deep-learning chambers"** |
| 11 | Real motion from a multi-phase CCTA | partly done on the office PC (29 Sep): there is no multi-phase data (CCTA = single-beat 75 %; only other phase = 45 % non-contrast calcium score). `scripts/phase_motion.py` + `tests/test_phase_motion.py` measure the 75→45 % outer-surface motion (validated: direction 0.95, magnitude −20 % without contrast). Result: surface motion is similar in size but not in direction to the model (agreement −0.22); the model's volume change is about twice the measured one. The model is **not** changed yet. Hospital request: README_3D.md, "Measured motion" |
| 12 | Unity / XR import of `heart_animated.glb` | import done and verified on the office PC (29 Sep, target PC VR): the GLB passes the Khronos validator with 0 issues (zero-length normals fixed in `mask_to_mesh`). Unity 6000.6.3f1 + glTFast 6.20.0: `unity/Assets/Heart/` (HeartController + HeartImportCheck, 38/38 checks, LV 31.60→14.29 mL = Python). Test project `C:\Users\MUBIN\UnityProjects\HeartXR`. **Next:** the VR rig (OpenXR + XR Interaction Toolkit); no OpenXR runtime is installed yet. README_3D.md, "Unity / XR" |
| 13 | Segmentation overlay in the DICOM tab | done on the office PC (29 Sep): `scripts/overlay.py` + sidebar control in `heart_dicom_viewer.py`; labels resampled in patient coordinates onto each slice; drawn only on the scan they were made on (same frame of reference + AcquisitionTime ≤ 2 s). `tests/test_overlay.py` (6). README_3D.md, "Segmentation overlay" |
| 14 | Synced slices, HU readout, MPR views in the DICOM tab | done on the office PC (29 Sep): `scripts/slice_view.py`. Axial / coronal / sagittal per window (MPR from a cached volume, radiological orientation, physical aspect). Hover readout (HU, structure, LPS mm) via an HTML canvas component, no new dependencies. **🔗 Sync position**: B–D follow window A by patient mm; other directions get a dashed crossing line. `tests/test_slice_view.py` (11). README_3D.md, "MPR, HU readout and position sync" |
| 15 | DICOM upload in the app | pending |
| 16 | Merge the v1/v2 viewers; fix v2 mask loading (masks are `*_mask.bmp` colour masks, not `*_gt.dcm`) | pending |

The details of 7–9 (methods, numbers, limitations) are in `README_3D.md`, in the sections *Epicardial fat* and *Coronary calcium*.

## 4. Step 10: notes for whoever picks it up
- **Goal.** *Accurate* chambers. Since 27 Sep 2026 `split_right_heart()` in `segment_heart.py` already separates the right atrium, right ventricle, pulmonary artery and SVC, plus the LV myocardium. It uses anatomy rules, because in this contrast timing right-heart blood (~130 HU) and myocardium (~150 HU) are too alike for HU rules. The outlines are approximate and meant for the animation; step 10 is about replacing them with deep-learning labels in the **same label IDs** (see below).
- **Candidate: TotalSegmentator.** It's built on nnU-Net. Its high-resolution heart-chamber task *may need a free academic licence key*; check the TotalSegmentator site before promising it. It runs on a CPU too, just slowly, while the GPU matters mainly for training your own models later (milestones M2 and M5).
- **Suggested design (still needs the user's agreement).**
  - Keep the tested Python 3.9 environment for the pipeline.
  - Put PyTorch (the CUDA build matching the driver) and TotalSegmentator in a **separate environment**, e.g. `.venv-dl`, run it as a command, and read back the NIfTI labels it writes. That avoids dependency conflicts with the lock file.
  - Keep the V1 rule-based path as the default or fallback, so the existing 30 tests keep guarding it.
  - Feed the new labels into `build_heart_model.py` (more meshes, same morph-target animation and glTF format).
- **Input.** `dataset/CT CCTA DL` (224 × 0.625 mm, contrast). The output to compare against is `output/3d_models/heart_labels.nii.gz`: 1 = LV blood pool, 2 = LA, 3 = aorta, 4 = myocardium, 5 = right atrium, 6 = right ventricle, 7 = pulmonary artery, 8 = SVC. `build_heart_model.py` animates all eight (the RV has its own contraction, the RA/SVC contract around the RA centre, and the PA distends with the aorta), so new labels in these IDs drop straight in.
- **Validation idea.** There's no chamber ground truth for this patient. Compare against V1 where the labels overlap (LV, LA, aorta: Dice and volumes), check the overlay visually, and look at whether the RV/RA volumes are plausible.

## 5. Working agreements and cautions
- **One step at a time.** Finish a step, report, then wait for the user's go.
- **Finish long final runs for real.** Deliverables such as the model file and reported numbers must come from the production code path (e.g. `python scripts/eat_benchmark.py --save-model`, about 50 min), not from cached or development results.
- **Privacy.**
  - The two images in `dataset/CT PatientReport/` have the **patient's full name and hospital ID burned into the footer**, although the DICOM Patient ID is anonymised.
  - The structured report names the approving physician, and the Rodriguez file names use patient initials.
  - Never repeat these details, and delete any scratch renders of those images. Redacting the footers was recommended but **not done**: the user has not decided yet, and it changes the dataset.
- **Calcium review belongs to a person.** A test review was saved and then deleted on purpose, so the app shows "Reviewed: —" until the user or a clinician reviews the 5 suggestions.
- **No credentials.** Don't type passwords or keys for the user; they set up logins, licence keys and SSH keys themselves.

## 6. Gotchas learned so far
- **DICOM decoding.** The TC_PTID DICOMs are JPEG Lossless (not JPEG2000, whatever the old manuals say). Read them with SimpleITK; plain pydicom can't decode them without extra plugins. The same goes for the RGB `CT PatientReport` images.
- **Slice order.** The EAT model expects **slice 0 = most superior**, like the Rodriguez files. The calcium-score series loads bottom-first in SimpleITK, so `eat_patient.py` and `calcium_score.py` flip it (`GetDirection()[8] > 0`). New inputs need the same check.
- **Rodriguez alignment.** The Rodriguez BMP images and masks are not pixel-aligned with the DICOM. `scripts/rodriguez.py` recovers each patient's scale and shift, and scoring happens on the annotation grid.
- **Two phases.** CT CALCIUM SCORE is at 45 % R-R; the CCTA (the 3D model) is at 75 %. They share a frame of reference but come from different breath-holds; a rigid registration shifts about 4.6 mm. SimpleITK's multi-threaded registration varies by about 0.1 mm between runs.
- **Byte-identical outputs.** Running the CLI scripts reproduces `output/` exactly. Calling `build_heart_model.main()` from Python passes bpm as `72.0`, not `72`, which changes the GLB/glTF bytes; regenerate `output/` with the CLI scripts.
- **Viewer opacity.** The "wall opacity" slider targets the myocardium mesh by name. Any new transparent mesh must keep that working.

## 7. CardioTwin XR roadmap (from 9 Oct 2026)
The user dropped the remaining 16-item steps and chose a new goal: a Unity heart like their "CardioTwin XR" concept image (realistic heart, mixed reality, hand control, layer and case panels). No VR headset yet, so Phase 0 (XR setup) waits.
- **Phase 1 order:** step 8 mesh quality **done 9 Oct** (README_3D.md, "Mesh quality") → 6 great vessels **done 9 Oct** (`scripts/vessels.py`, README_3D.md "Great vessels"; run it before `build_heart_model.py`) → 9 fat layer **done 9 Oct** (`fat_layer` in `build_heart_model.py`: the CCTA's own fat that touches the heart near the mapped EAT, 60.3 mL display layer; measured EAT stays 73.4 mL; README_3D.md "Epicardial fat layer") → 10 motion **done 10 Oct** (blended per-structure motion, so shared surfaces stay joined; the wall keeps its volume; the atrial roof and the arch are held; the IVC is held; README_3D.md "Motion") → 11 case package **done 10 Oct** (`scripts/case_package.py` → `output/case_package/DEMO-001/`; Unity `HeartCase.cs` + `CasePackageCheck.cs`, 22/22; README_3D.md "Case package") → 7 coronaries and 5 DL chambers (need the TotalSegmentator licence, still not set).
- Then Phase 2: 12 realistic materials **done 10 Oct** (`unity/Assets/Heart/Shaders/`, `HeartMaterials.cs`; import check 55/55; README_3D.md "Realistic materials") → 13 lighting **done 12 Oct** (`HeartContactShadow.cs`, room-style lighting, desk in the demo scene; import check 58/58; Linear colour space recommended, the user decides; README_3D.md "Lighting and contact shadow"). Test project switched to Linear colour space (12 Oct). Phase 3: 14 interaction **done 12 Oct** (`HeartManipulator.cs` input-agnostic rotate / zoom / move / reset, resting on the desk; `HeartMouseInput.cs` for desktop; import check 62/62; README_3D.md "Interaction") → 15–20 interface **done 12 Oct** (`HeartUI.cs` uGUI toolbar / Layers / Case / slice controls with a 2D CT view / labels; `HeartSlice.cs` + `Heart/Slice Cap` cut with the case package's CT on the cut face; `HeartPicker.cs`; uGUI 2.6.0 added to the project; interface check 20/20, import check 63/63; README_3D.md "Interface"). Phase 3 complete. Phase 4 **done 12 Oct**: 21 look (title bar, notice, icons, frosted glass), 22 headset performance (`heart_light.glb` in the case package, 86k triangles; Headset quality; benchmark 8–10 ms per stereo frame on the RTX 2060), 23 comfort (`HeartStage`: Recentre seated / standing, lazy-follow panels), 24 case picker (runtime loading with glTFast legacy animation). Interface check 29/29, import check 64/64; README_3D.md "Product polish". Next: Phase 5 (validation: device tests, expert review, usability study); still blocked: steps 7 and 5 (TotalSegmentator licence).
- The user may have the Unity project `C:\Users\MUBIN\UnityProjects\HeartXR` open in the Editor. Batch-mode checks then fail on it, so run them in a scratch copy, and don't copy files into the open project unasked.

## 8. Status 12 Oct 2026 (session stopped at the usage limit)
- **TotalSegmentator licence:** set by the user (free academic licence, stored in `C:\Users\MUBIN\.totalsegmentator`).
  - The key is personal: don't copy it into files.
  - The Claude app is MSIX-packaged. Its private AppData hid uv's Pythons from the user's own shells, so they were copied to `C:\Users\MUBIN\uv-python`, and both venvs' `pyvenv.cfg` now point there (old files in `pyvenv.cfg.bak`).
- **Step 5, deep-learning chambers: done in the pipeline.**
  - `segment_heart_dl.py` ran `heartchambers_highres` (317 s). It now prefers `~/venvs/heart3d-dl` over PATH, where a CPU-only TotalSegmentator sits in the user's system Python 3.10.
  - `build_heart_model.py --chambers auto|dl|rules` builds from the DL labels when present: `model_labels`, `chamber_source`, `lv_long_axis`, `vessels.merge(deep=True)`.
  - New model: LV 52.3 mL at rest, EDV 60.0 / ESV 22.8, EF 62 %.
  - The atrial kick was balanced (`atrial_scale` 0.92, `a_wave` 0.07, `rv_atrial_systole`), and the wall keeps its volume in atrial systole.
  - The case package was re-exported, with provenance per structure.
  - Checks: Unity 64/64, case 22/22, interface 29/29 (scratch clone); Python 86 tests pass.
  - Documented on 13 Oct (README "Deep-learning chambers", step 5).
- **Step 7, coronaries: segmentation done 12 Oct; integrated 13 Oct 2026** (README "Coronary arteries").
  - `scripts/coronaries.py` → `output/3d_models/coronaries_dl.nii.gz` (CT grid 0.377 mm, cropped; 2.17 mL, 12 pieces). TotalSegmentator predicts at 0.7 mm, hence 0.7 mm steps in the mask.
  - `build_heart_model.py`: label 13 "Coronary arteries" on the 1 mm grid (background only, before the fat layer); own mesh from the fine mask (`coronary_mesh`: σ 0.35 mm, level 0.4, 30 Taubin half-steps; 52k triangles, 2.3 mL); no motion of its own (`CARRIED`, like the fat).
  - Unity: `HeartMaterials` → oxygenated; `HeartImportCheck.k_Optional` += "Coronary arteries"; `CasePackageCheck` and `test_case_package` use a 0.75 mm inset for it.
  - Checks: Khronos 0 issues (347k triangles, light 86k); Unity 66/66, case 23/23, interface 29/29, in the scratch clone **and in the user's `HeartXR`** (installed 13 Oct: GLB, case package, the three changed scripts); Python 89 tests pass.
- **Next:** Phase 5 (validation). The import check run also rebuilt `HeartXR`'s `Assets/Heart/HeartDemo.unity` with the new model, so the user only opens it and presses Play.
- **Stray folder:** `output/3d_models/` holds an empty Unity project (Assets, Library 105 MB, …, `3d_models.slnx`) made on 12 Oct 11:47, probably from opening that folder in Unity Hub. Offered to remove it; not deleted.
