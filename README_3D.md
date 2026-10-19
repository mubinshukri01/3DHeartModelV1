# 3D Heart Model V1 — Segmentation, Cardiac Cycle, glTF, Streamlit

UTM-Claude Research Project · Assoc Prof Ts Dr Rohayanti Hassan · Thesis Objective 1

## Quick start

Run from the project folder (the one with this file); the project can live anywhere.

```
uv venv .venv --python 3.9
uv pip install --python .venv\Scripts\python.exe -r requirements_3d.lock.txt
.venv\Scripts\python scripts\segment_heart.py        # ~8 s  -> labels + QC overlay
.venv\Scripts\python scripts\build_heart_model.py    # ~2 s  -> animated GLB/glTF + STL/OBJ
.venv\Scripts\streamlit run scripts\heart_dicom_viewer.py
```

Without uv, create the venv with `python -m venv .venv` (Python 3.9) and use `.venv\Scripts\pip install -r requirements_3d.lock.txt`. On an ARM laptop (e.g. Snapdragon) use `--python cpython-3.9-windows-x86_64-none`, because SimpleITK and scikit-image have no Windows-ARM wheels for 3.9. The x86 emulation makes segmentation take ~35 s instead of ~8 s.

In the app, open the **🫀 3D Heart Model** tab. The **▶ Run pipeline** button runs both steps from inside the app. It only appears when `dataset/` has CT series, so it is hidden on the server.

## Dependencies

| File | Purpose |
|---|---|
| `requirements_3d.txt` | Direct dependencies with loose ranges. Edit this one. |
| `requirements_3d.lock.txt` | Exact versions the pipeline was tested with (Python 3.9). Install from this. Regenerate after editing the file above: `uv pip compile requirements_3d.txt --python-version 3.9 -o requirements_3d.lock.txt`, then re-run the tests. |
| `requirements_dev.txt` | Lock file + pytest |

## Tests

```
uv pip install --python .venv\Scripts\python.exe -r requirements_dev.txt
.venv\Scripts\python -m pytest -v
```

`tests/test_pipeline.py` runs both pipeline steps into a temporary folder (the real `output/` is untouched). It checks the volumes, the label-map overlap with `tests/reference/heart_labels.nii.gz` (Dice ≥ 0.99), the LV cycle (EDV 37.3 / ESV 15.1 mL, EF ≈ 59 %; rule-based labels), the output files and the GLB structure. It takes ~1.5 min. A 20 HU change to one threshold already fails it. Without the dataset only the three unit tests run.

## Tuning the segmentation

All thresholds are named constants at the top of `scripts/segment_heart.py`:

| Constant | Default | Meaning |
|---|---|---|
| `SMOOTH_SIGMA` | 1.5 | Gaussian smoothing (after a 3-voxel median) before any threshold |
| `CONTRAST_LOWER` / `CONTRAST_UPPER` | 300 / 700 HU | Iodinated blood pool (LV, LA, aorta) |
| `ENVELOPE_LOWER` / `ENVELOPE_UPPER` | 110 / 700 HU | Whole heart (contrast-enhanced myocardium + right heart) |
| `BONE_FILL_HU` / `BONE_HU` | 600 / 900 HU | Bone mask (the vertebral body is filled in) |
| `EXCLUDE_MIN_VOX` | 5000 | Other bright blobs larger than this (descending aorta, IVC) are cut away |
| `EROSION_RADIUS` | 4 | Breaks thin connections to the liver and chest wall. Raise if the mask leaks, lower if it has holes. |
| `LV_APEX_SLAB` / `AORTA_TOP_SLAB` | 15 / 8 slices | Seed regions for the LV (apex) and aorta (top) |
| `MITRAL_CUT_FACTOR` | 1.1 | Position of the LV/LA split plane, in LA radii |

After a change, check `output/visualizations/segmentation_overlay.png`. If the change is intended, update the reference in the tests.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `segment_heart.py` | Loads **CT CCTA DL** (224 × 0.625 mm, contrast), resamples it to 1 mm, and runs a training-free HU segmentation: whole heart (>110 HU, liver/spine/descending aorta cut away), bright left blood pool (300–700 HU), then **LV / LA / Aorta** (edge watershed for the aortic valve, a long-axis plane at the mitral annulus, and an LVOT correction). The remainder is **myocardium + right heart**. |
| 1b | `eat_patient.py` | *Optional.* Measures **epicardial fat (EAT)** on the non-contrast **CT CALCIUM SCORE** series with the trained EAT model, then maps the fat onto the 1 mm CCTA grid with a rigid registration. See [Epicardial fat](#epicardial-fat-eat). |
| 1c | `calcium_score.py` | *Optional, after 1b.* **Agatston coronary calcium score** on CT CALCIUM SCORE: automatic lesion suggestions in the coronary territory, then reader review in the app. See [Coronary calcium](#coronary-calcium-agatston). |
| 1-DL | `segment_heart_dl.py` | *Optional, after 1; needs a TotalSegmentator licence for the chambers.* **Deep-learning chambers** with TotalSegmentator, in the same label ids, compared with the rules. Step 2 builds the model from them when they exist. See [Deep-learning chambers](#deep-learning-chambers-cardiotwin-xr-roadmap-step-5-12-oct-2026). |
| 1e | `coronaries.py` | *Optional, after 1; needs the licence.* **Coronary arteries** from TotalSegmentator's `coronary_arteries` task, meshed by step 2 at the CT's finer resolution. See [Coronary arteries](#coronary-arteries-cardiotwin-xr-roadmap-step-7-13-oct-2026). |
| 1d | `vessels.py` | *Optional, after 1; needs TotalSegmentator.* **Great vessels** (SVC, IVC, pulmonary veins, descending aorta, LA appendage, aortic root) from the open `total` task, merged into the model by step 2. See [Great vessels](#great-vessels-cardiotwin-xr-roadmap-step-6-9-oct-2026). |
| 11 | `phase_motion.py` | *Optional, after 1.* **Measured motion** between the 75 % CCTA and the 45 % calcium-score scan (rigid + B-spline registration), compared with the procedural model at 45 %. Does not change the model. See [Measured motion](#measured-motion-step-11). |
| 14 | `slice_view.py` | DICOM tab: coronal / sagittal views, HU readout on hover, and position sync between windows. See [MPR, HU, sync](#mpr-hu-readout-and-position-sync-in-the-dicom-tab-step-14). |
| 13 | `overlay.py` | Segmentation overlay in the app's DICOM tab (sidebar **🧩 Segmentation overlay**). See [DICOM overlay](#segmentation-overlay-in-the-dicom-tab-step-13). |
| 12 | `unity/Assets/Heart/` | *Optional, Unity.* `HeartController.cs` (plays the GLB's cardiac cycle in Unity; heart rate, opacity, visibility, phase) and `Editor/HeartImportCheck.cs` (38 import checks, demo scene, preview render). See [Unity / XR](#unity--xr-step-12). |
| 2 | `build_heart_model.py` | Builds marching-cubes meshes with Taubin smoothing, adds three **glTF morph targets** (ventricular systole, atrial systole, aortic distension), and a looping `CardiacCycle` animation with Wiggers-diagram timing. If step 1b has run, the fat is shown as a display layer of the CCTA's own fat that moves with the heart (step 9). |
| 2b | `case_package.py` | **Case package for the XR app** (`output/case_package/<case id>/`): the model, a case card (`case.json`: structures with volume, colour and provenance, the fat and calcium measurements, the simulated motion) and the CT and labels as raw volumes with the model → voxel transforms. No patient identifiers. Run by **▶ Run pipeline** after step 2. See [Case package](#case-package-cardiotwin-xr-roadmap-step-11-10-oct-2026). |
| 3 | `heart_3d_tab.py` | Streamlit tab: an embedded three.js viewer (toggle structures, heart-rate slider, wall opacity, ECG trace, phase label), volumes, EAT, calcium review, downloads, QC figures. **▶ Run pipeline** runs 1, then 1b and 1c (when the calcium-score series and the EAT model exist), then 2. |

### Motion model (procedural)
The dataset contains **one cardiac phase** (CCTA at 75 % R-R, i.e. diastasis), so that shape is the rest pose. Systole is modelled with:
- **Cavity contraction.** The LV cavity contracts radially. Its radius is fitted to this patient's LV at each height and angle around the long axis.
- **Wall.** The wall keeps its volume (myocardium is incompressible), so it thickens as it shortens.
- **Shortening.** 15 % base-to-apex.
- **Atria and great arteries stretch.** Their floor follows the AV plane, while their top is held: the atrial roof by the veins (40 mm above the mitral plane), the great arteries at the arch (70 mm).
- **Twist.** 10° LV twist.
- **Atrial systole.** 8 % atrial contraction, with a 7 % A-wave filling of both ventricles (step 5).
- **Arteries.** 8 % distension.
- **Each structure has its own motion** (`motion_targets`):
  - the RV free wall contracts towards the RV axis;
  - the RA and SVC contract around the RA centre;
  - the IVC is held by the diaphragm;
  - the descending aorta only distends.
- **Neighbours blend** over a few millimetres (`motion_blend`), so shared surfaces move together. The fat and the coronary arteries move with what they lie on.

The resulting LV blood-pool curve gives **EF ≈ 62 %** with the deep-learning chambers (59 % with the rules) (see `output/visualizations/cardiac_cycle.png`). All parameters are in `CardiacMotion(...)`. See [step 10](#motion-cardiotwin-xr-roadmap-step-10-10-oct-2026) for what changed and why.

**Real motion upgrade:** with a multi-phase (4D, 0–90 % R-R) CCTA export, segment each phase and fill the same morph-target slots with the per-phase displacements. Registering phase *k* to the 75 % mesh would do this, for example with SimpleITK B-spline registration. The viewer, GLB format and Unity pipeline stay unchanged.

## Outputs (`output/3d_models`)
| File | Use |
|---|---|
| `heart_animated.glb` | Single file for web and **Unity** (glTFast / UniGLTF import the morph targets as blend shapes and the animation as a clip) |
| `heart_animated.gltf` + `.bin` | Same model, human-readable JSON |
| `heart_{lv,la,aorta,myocardium}.stl/.obj` | Static meshes for 3D printing / CAD |
| `heart_viewer.html` | Standalone web viewer. Run `python -m http.server` in this folder and open `http://localhost:8000/heart_viewer.html` |
| `heart_labels.nii.gz`, `heart_ct_1mm.nii.gz` | Label map + CT (for 3D Slicer / ITK-SNAP QC) |
| `segmentation_stats.json`, `animation_info.json` | Volumes, long axis, LV volume curve |
| `heart_eat.stl/.obj` | Epicardial-fat display layer as a closed shell (step 1b; see step 9) |
| `heart_labels_dl.nii.gz`, `dl_segmentation.json` | Deep-learning chambers on the 1 mm grid, and their comparison with the rules (step 1-DL) |
| `coronaries_dl.nii.gz`, `coronaries_dl.json`, `heart_coronaries.stl/.obj` | Coronary arteries on the CT's own grid, their stats, and their closed mesh (step 1e) |
| `eat_stats.json` | EAT volume and attenuation, model and registration details |
| `eat_cac.nii.gz`, `eat_labels.nii.gz` | EAT mask on the calcium-score grid (measured) and on the CCTA grid (display) |
| `calcium_stats.json` | Automatic Agatston / volume score, suggested lesions, criteria, the report's value |
| `calcium_review.json` | The reader's review, written by the app: which suggested lesions are coronary |
| `../case_package/<case id>/` | XR case package (step 2b): `heart.glb`, `case.json`, `ct.raw`, `labels.raw` |

Coordinates: glTF metres, +X = patient left, +Y = superior, +Z = anterior. The model is centred on the heart.

## Epicardial fat (EAT)

EAT is the fat inside the pericardium. It is the focus of the research roadmap (`EAT_Milestone_Workflow.docx`), and here it is the groundwork for milestone M2.

### Method (`scripts/eat_segment.py`)
The method works on **non-contrast** CT and has two stages:
1. **Rules (training-free).** Find the heart as the largest central soft-tissue blob and outline it in each slice (convex hull). Near the apex and base, where the heart mask thins out, the outline is borrowed from neighbouring slices. The outline is widened by 15 mm. This candidate zone contains about 100 % of the EAT on the benchmark, plus mediastinal, paracardial and abdominal fat.
2. **Classifier.** A gradient-boosted classifier (scikit-learn) scores every candidate fat voxel from 19 features: distances to the heart outline, lung and bone; local density (liver vs blood); position around the heart; and slice level. Voxels scoring ≥ 0.4 are EAT.

The trained model is `models/eat_classifier.joblib` (0.5 MB, trained on all 18 benchmark patients).

### Validation (`scripts/eat_benchmark.py`, Rodriguez dataset)
`dataset/Rodriguez` holds 20 non-contrast calcium-score CTs (Philips iCT 256 / Siemens Sensation 64, 3 mm slices) with expert masks.
- **Ground truth.** In the masks, red = EAT, green = mediastinal fat, and blue = the drawn pericardium line, which is ignored in scoring.
- **Registration.** The BMP images and masks are *not* pixel-aligned with the DICOM. Each patient was shifted, and scans with another pixel size were resampled to 0.3516 mm. `scripts/rodriguez.py` recovers the scale and shift from the images themselves (correlation 1.000 when not resampled, about 0.9 when resampled) and scores on the annotation grid.
- **Excluded patient.** TJes has a single slice and is left out, which leaves 18 patients.

Every patient is scored by a model trained on the other 17 (leave-one-patient-out):

| Method | Dice (mean ± SD) | Median (range) | Volume r | Volume bias (95 % LoA) |
|---|---|---|---|---|
| Rules only (fat ≤ 6 mm from the heart outline) | 0.652 ± 0.087 | 0.657 (0.51–0.78) | 0.75 | +78 mL |
| **Rules + classifier** | **0.805 ± 0.055** | **0.819 (0.72–0.89)** | **0.91** | **+1.0 mL (−34 to +36)** |

The mean expert EAT volume is 103 mL. The threshold 0.4 was chosen on these cross-validated predictions because it gave unbiased volumes; in the development run, 0.5 gave Dice ≈ 0.80 with a bias of ≈ −8 mL. The roadmap's M2 target is Dice ≥ 0.85.

Weakest cases:
- **JMir** (0.72): thick paracardial fat is partly called EAT.
- **CLis** (0.73): the only full-chest field of view. A thin anterior EAT band is missed, as are the lowest heart slices.
- **ISou** (0.73): the EAT is under-segmented.

Figures and per-patient numbers are in `output/eat_benchmark/`. To reproduce, run from `scripts/`:
```
python eat_benchmark.py --rules-only      # ~7 min
python eat_benchmark.py --save-model      # ~50 min on a laptop CPU; retrains models/eat_classifier.joblib
```

### This patient (`scripts/eat_patient.py`)
- **Result.** TC_PTID, CT CALCIUM SCORE (2.5 mm, 45 % R-R): **EAT 73.4 mL, mean attenuation −78.5 HU**, using the standard fat window −190 to −30 HU.
- **Mapping to the 3D model.** For display, the mask is moved onto the 75 % CCTA grid. The two series share a frame of reference but were acquired in different breath-holds, so a rigid registration corrects the offset: about 4 mm shift and 2° rotation. The heart phases also differ (45 % vs 75 %), so the mapped mask doesn't fit the CCTA heart well. The 3D model therefore shows a display layer of the CCTA's own fat, guided by the mapped mask (see [step 9](#epicardial-fat-layer-cardiotwin-xr-roadmap-step-9-9-oct-2026)). The volume is always the one measured on the calcium-score scan.
- **Checks.** See `output/visualizations/eat_overlay.png` (the same heights in both scans) and the 🟡 QC panel in the app.

## Coronary calcium (Agatston)

`scripts/calcium_score.py` scores the non-contrast **CT CALCIUM SCORE** series with the standard Agatston rules:
- **Lesion.** A connected region ≥ 130 HU of at least 1 mm² on a slice.
- **Score.** Area (mm²) × weight of the lesion's peak HU: 1 for 130–199, 2 for 200–299, 3 for 300–399, 4 for ≥ 400. The total is scaled by slice spacing / 3 mm (Agatston's reference slices; × 0.83 for this 2.5 mm series).
- **Volume score.** The lesion volume in mm³.
- **Category.** CAC-DRS 0–3.

Only *coronary* calcium counts. Like the scanner's own software, the workflow is **semi-automatic**:
1. **Automatic suggestion.** A lesion is suggested when it lies in the heart zone (the pericardial outline from `eat_segment.py`, + 5 mm) and within 3 mm of epicardial fat, where the coronary arteries run. Bone-sized regions and spots in the blood pool or myocardium (valves, noise) are not suggested.
2. **Reader review.** In the app (🦴 *Coronary calcium*), each suggestion is shown as a thumbnail with a checkbox. The reader unticks what is not coronary and saves; the result goes to `calcium_review.json`. Re-running the script clears an old review, because lesion ids change.
3. **Report value.** If the dataset has a DICOM structured report, its Agatston value is shown alongside.

**TC_PTID:**
- **Automatic suggestion:** 5 lesions, **17.5**. All are tiny (1.4–6.9 mm²), faint (141–203 HU) specks at tissue edges, which pass the standard criteria.
- **Excluded automatically:** 11 other bright spots (10 away from the fat, 1 bone).
- **Radiology report:** **0 Agatston**. The scanner's scoring report shows 0 for every artery.
- **After review:** rejecting the five specks gives **0**, matching the report.

**Checks** (`tests/test_calcium.py`, 13 tests):
- The weight bands, area rule and scaling are tested on synthetic slices.
- A synthetic calcification placed in the patient's epicardial fat is suggested and scored exactly.
- The same lesion placed deep in the blood pool (valve-like) is not suggested.

**Limitations:**
- Only one real case (score 0) is available, so the automatic suggestion has not been validated on patients who have calcium.
- Mitral-annulus and aortic-root calcium next to fat would be suggested and must be rejected by the reader.
- A fully automatic scorer would need labelled scans (for example the IJN dataset of milestone M3).

## Deep-learning chambers (CardioTwin XR roadmap, step 5, 12 Oct 2026)

`scripts/segment_heart_dl.py` runs [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (nnU-Net) on CT CCTA DL at native resolution. It maps the result onto the 1 mm grid and the label ids of `segment_heart.py`. **When these labels exist, `build_heart_model.py` builds the model from them** (`--chambers auto`, the default); `--chambers rules` uses the rule-based labels, and `--chambers dl` insists on the deep-learning ones. The rule-based labels stay the fallback and the reference of `tests/test_pipeline.py`.

| Id | Label | TotalSegmentator task : class |
|---|---|---|
| 1 | LV blood pool | `heartchambers_highres` : `heart_ventricle_left` |
| 2 | LA blood pool | `heartchambers_highres` : `heart_atrium_left` |
| 3 | Aorta | `heartchambers_highres` : `aorta` (open `total` task when no licence), ascending part only |
| 4 | Myocardium | `heartchambers_highres` : `heart_myocardium` |
| 5 | Right atrium | `heartchambers_highres` : `heart_atrium_right` |
| 6 | Right ventricle | `heartchambers_highres` : `heart_ventricle_right` |
| 7 | Pulmonary artery | `heartchambers_highres` : `pulmonary_artery` |
| 8 | Superior vena cava | `total` : `superior_vena_cava` (fills only voxels no chamber claimed) |

**Setup (once).** TotalSegmentator needs Python ≥ 3.10 and PyTorch, so it gets its own environment. Keep it outside OneDrive, because it is about 5 GB:
```
uv venv %USERPROFILE%\venvs\heart3d-dl --python 3.12
uv pip install --python %USERPROFILE%\venvs\heart3d-dl\Scripts\python.exe --torch-backend auto torch TotalSegmentator
```
- The scripts find TotalSegmentator via `--totalseg`, the `HEART3D_TOTALSEG` environment variable, `%USERPROFILE%\venvs\heart3d-dl`, then `PATH`, in that order. A TotalSegmentator on `PATH` may be a CPU-only install in another Python (the office PC has one), so the GPU environment comes first.
- **Licence.** `heartchambers_highres` and `coronary_arteries` need a licence, which is free for non-commercial use (academic licence page on the TotalSegmentator site). Set it with `%USERPROFILE%\venvs\heart3d-dl\Scripts\totalseg_set_license -l <key>`. Without it only the open `total` task runs, so you get the aorta and SVC but no chambers.
- **Usage statistics.** TotalSegmentator reports each run (task, options, versions, licence number; no image data) unless `"send_usage_stats": false` is set in `%USERPROFILE%\.totalsegmentator\config.json`. It is off on the office PC.

**Run** (after `segment_heart.py`):
```
.venv\Scripts\python scripts\segment_heart_dl.py        # ~7 min on an RTX 2060 with the licence (both tasks)
```
Output: `output/3d_models/heart_labels_dl.nii.gz`, `dl_segmentation.json` (volumes, Dice vs the rules, sources, versions), and `output/visualizations/dl_vs_rules_overlay.png` (rules above, deep learning below).

**TC_PTID with the licence (12 Oct 2026):**

| Structure | Rules (mL) | Deep learning (mL) | Dice |
|---|---|---|---|
| LV blood pool | 33.0 | 52.6 | 0.77 |
| LA blood pool | 61.1 | 47.4 | 0.83 |
| Aorta | 23.7 | 27.2 | 0.91 |
| Myocardium | 102.0 | 70.5 | 0.55 |
| Right atrium | 23.3 | 57.9 | 0.55 |
| Right ventricle | 48.8 | 67.4 | 0.80 |
| Pulmonary artery | 29.9 | 40.6 | 0.68 |
| Superior vena cava | 3.6 | 3.9 | 0 |

- **In the overlay the deep-learning chambers look anatomically better.** The LV includes the trabeculae (the rules took the bright blood only), the RA / RV split follows the tricuspid valve, and "Myocardium" is the true LV wall (the rule label also held thin sheets around the LA and the aortic root).
- **The SVC** is TotalSegmentator's from the open task, in the right place (the rule SVC lay in the right atrial appendage).
- These are visual checks against the rules, not a validation against expert labels.

**The model with the deep-learning chambers:**
- **Vessels.** `vessels.merge(deep=True)` lets the vessels fill background only (`REPLACES_DL`), plus the venae cavae's junction with the RA, because every chamber label is now anatomical.
- **LV long axis** from the labels (`lv_long_axis`): the base is the LV within 2 mm of the LA (the mitral valve), the apex the mean of its farthest 0.5 %.
- **LV** 52.3 mL at rest; procedural **EDV 60.0 / ESV 22.8 mL, EF 62 %**. In Unity 52.32 → 22.77 mL, equal to Python.
- **The atrial kick was rebalanced** on these chambers. With the old settings the atria lost 32 mL but the ventricles gained only 14 mL:
  - atrial contraction 15 → 8 % (`atrial_scale` 0.92), A-wave filling 5 → 7 % (`a_wave` 0.07);
  - the RV fills too (`rv_atrial_systole`; it barely changed before);
  - the LV wall keeps its volume in the A-wave as in systole (it grew 12 %).
- **Result:** whole heart −3.9 % at peak systole and −0.6 % in the atrial kick; wall −2.9 % / +0.7 %; shared surfaces still meet (0 mm seams). With the rule labels the EDV moved 36.4 → 37.3 mL (`tests/test_pipeline.py`).
- GLB extras record the source: `"chambers": "deep learning (TotalSegmentator heartchambers_highres)"`, and the case package's provenance says so per structure.

**Earlier, without the licence (open task only):** aorta 29.6 vs 23.7 mL (Dice 0.88; the descending aorta is dropped, keeping the part within 3 mm of the LV, or else the most anterior part); whole heart (open `heart` class, which also covers the pulmonary artery) 340 vs 325 mL, Dice 0.77.

**Gotchas:**
- **No `roi_subset`.** The open task's `roi_subset` shortcut crops with a 6 mm model; on this small cardiac field of view it returned an empty SVC. The script runs the full task instead.
- **GPU memory.** The open task peaked at 4.7 GB on the 6 GB RTX 2060. Close GPU-heavy apps before a run; `--device cpu` works too, but slowly.
- **Windows workers.** A script that calls TotalSegmentator's Python API needs an `if __name__ == "__main__":` guard. nnU-Net starts its workers with `spawn`, so without the guard every worker re-runs the script and RAM runs out. `segment_heart_dl.py` avoids this by calling the `TotalSegmentator` command.

**Tests:** `tests/test_dl.py` (7 synthetic tests). They cover the aorta trimming, the id mapping, the SVC never overwriting a chamber, the open-task-only case, and the executable lookup and its order. `tests/test_motion.py` covers the chamber source, the LV long axis from the labels and the wall in the atrial kick; `tests/test_vessels.py` the vessel merge with the deep-learning chambers.

## Measured motion (step 11)

**Data.** Real motion needs the heart at several phases, but TC_PTID has only two:
- **75 % R-R:** CCTA, Lungs, Mediastinum and SSFreeze, all reconstructions of one contrast scan.
- **45 % R-R:** the non-contrast CT CALCIUM SCORE, a separate scan and breath-hold about 6 minutes earlier.

The dose report shows the CCTA was a single-beat "Stationary Acquisition" (140 mm coverage, 0.28 s rotation, 0.78 s beam-on). The raw data may cover a window of phases around 75 %, but scanners keep raw data only for a limited time, and this scan is from May 2023. A true 0–90 % series for this patient is almost certainly not recoverable (see [Hospital request](#what-to-ask-the-hospital-for)).

**Method** (`scripts/phase_motion.py`):
1. **Rigid registration** (`eat_patient.register`) removes the breath-hold offset.
2. **B-spline registration** (20 mm control grid, Mattes mutual information, heart + 15 mm, pyramid 4-2-1 mm) gives the deformation from 75 % to 45 %. Mutual information copes with contrast vs no contrast.
3. **Comparison.** On the heart's outer surface the measured displacement is compared with the procedural morph targets at 45 %, using weights W(45 %) − W(75 %). The model's weights at 75 % are not exactly 0 (aortic distension 0.14), so the difference is what counts. The best-fitting rigid motion is removed from **both** sides first, because a breath-hold shift cannot be told apart from rigid motion of the heart. The whole-heart volume change (mean Jacobian) is compared too.
4. **`--validate`** applies the model's own 45 % motion to the CCTA and measures it back: with contrast, with the contrast removed (blood and myocardium set to ~45 HU, like the real 45 % scan), and with no motion at all (the noise floor).

Without contrast, blood and myocardium look alike at 45 %, so **only the outer surface** (against fat and lung) is measurable. The chamber walls inside are not.

**Run:**
```
.venv\Scripts\python scripts\phase_motion.py             # ~2.5 min
.venv\Scripts\python scripts\phase_motion.py --validate  # + three known-answer checks, ~12 min
```
Output: `output/3d_models/phase_motion.json` (+ `phase_motion_validation.json`), the B-spline `phase_motion_bspline.tfm`, and `output/visualizations/phase_motion.png`. The top row of the figure is the rigidly aligned 45 % scan with the heart outline at 75 % (green) and registered to 45 % (red). The bottom row is the surface motion, measured vs model.

**Validation** (`--validate`, the model's 45 % motion applied to this CCTA; outer surface, non-rigid part; re-run 10 Oct 2026 with the step-10 motion):

| Known motion | True | Measured | Mean error (95th pct) | Direction agreement | Whole-heart volume ratio (true → measured) |
|---|---|---|---|---|---|
| Model motion, with contrast | 1.95 mm | 1.71 mm | 0.53 (1.29) mm | 0.97 | 0.998 → 0.993 |
| Model motion, contrast removed | 1.95 mm | 1.55 mm | 0.64 (1.59) mm | 0.96 | 0.998 → 1.009 |
| No motion, contrast removed | 0 | 0.07 mm | 0.07 (0.14) mm | – | 1.000 → 1.002 |

- **Direction is recovered well.** Without contrast, as in the real 45 % scan, the direction of surface motion comes back reliably (agreement 0.96; 1 = identical, 0 = unrelated, −1 = opposite).
- **Magnitude is underestimated by about 20 %.**
- **The noise floor is tiny.** When nothing moves, the method measures almost nothing (0.07 mm). This is an idealised floor, because the real scans also differ in slice thickness (2.5 vs 0.625 mm), noise and breath-hold.
- **The volume ratio is good to about ±0.01.** That's the error on the known motion. Three runs on the real scans gave 0.963, 0.976 and 0.969, because ITK's multithreaded registration doesn't repeat exactly.
- The known motion is the model's generic ventricle/atrium field everywhere (`procedural(..., generic=True)`), not the per-structure motion.

**TC_PTID, 75 % → 45 %** (outer surface, rigid part removed on both sides):

| | Measured (10 Oct run) | Model, step 11 (29 Sep) | Model, step 10 (10 Oct) |
|---|---|---|---|
| Surface motion, mean | 3.44 mm | 2.63 mm | 2.19 mm |
| Direction agreement with the model | | **−0.22** | **−0.23** |
| Longitudinal, apex / base (+ = towards base) | −1.0 / +1.6 mm | +1.9 / −2.7 mm | +1.8 / −2.7 mm |
| Whole-heart volume at 45 % vs 75 % | −3.1 % (runs: −2.4 to −3.7 %) | −8.2 % | **−1.8 %** |

- **How the model's volume is measured now.** Since step 10 it is the volume inside the outer surface under the model's motion. The blended field is continuous, so this is the model's mean Jacobian over the heart, like the measurement. On 29 Sep it was the sum of the per-structure meshes.
- **Rigid offset.** The rigid step shifted the 45 % scan by (1.6, −2.1, −3.7) mm and rotated it 2°, the same breath-hold offset as in the EAT step.

**Reading:**
- **The whole-heart volume change now matches** within the method's error (−1.8 % vs −3.1 %; it was −8.2 %). The cause of the mismatch was the one suspected here: the model dragged the atria and great vessels towards the apex with the valve plane. Step 10 holds the atrial roof and the arch, so the valve plane moves *inside* a nearly constant outline, and the wall now keeps its volume.
- **The direction still doesn't match.** The measurement shows the base moving *away* from the apex. In every heart the AV plane descends towards the apex during ejection, so this more likely reflects the comparison than the motion:
  - **This is one patient and two separate breath-holds, and the heart rates are unknown.** At a slower rate, 45 % R-R is already early diastole, when the base moves back up.
  - Non-rigid breathing effects cannot be separated from cardiac motion.
  - The model was not tuned to this direction. The chamber walls inside remain unmeasured.

**Tests:** `tests/test_phase_motion.py` (5 synthetic tests). They cover the weights at a phase, rigid-motion removal, the glTF/LPS conversion, contrast removal, and recovering a known 2 mm contraction of a phantom heart (error < 0.6 mm; 0.38 mm measured).

### What to ask the hospital for
Real motion needs the **same contrast acquisition reconstructed at several cardiac phases**:
- **TC_PTID.** Ask whether the raw (projection) data of the CCTA (22 May 2023, protocol "5.19 Coronary SmartPrep - DLIR", series 301) is still archived. If it is, ask for reconstructions at every phase the data allows, every 5 %, with the same settings as series 301: 0.625 mm, STANDARD kernel, DLIR, same field of view and centre. This is unlikely after this long.
- **Existing multi-phase scans, no extra scanning.** Retrospectively gated cardiac CTs are often reconstructed at 0–90 % R-R already, for example many TAVI-planning or functional cardiac CTs. Anonymised examples of these would let the pipeline measure real motion without any new radiation.
- **New scans.** Wider ECG gating (retrospective, or prospective with wide padding) increases the radiation dose. Whether to use it is a clinical and ethics decision, not something to request for the model alone.
- **Format.** One DICOM series per phase, with `NominalPercentageOfCardiacPhase` filled in, identical slice thickness, kernel and field of view across phases. Anonymise them, including burned-in text in any screenshots or report images. The heart rate during the scan is useful too.

## Segmentation overlay in the DICOM tab (step 13)

In the app's **🪟 DICOM Multi-View** tab, the sidebar has a **🧩 Segmentation overlay** control that applies to all windows:
- **Layers:** *Heart (rules)*, *Heart (deep learning)* and *Epicardial fat*. Each appears only once its output file exists. For now the deep-learning layer has only the aorta and SVC (see step 10).
- **Opacity** slider.
- **Legend** under each window: the structures on the slice on screen, with their area in cm².

**How it works** (`scripts/overlay.py`). A label volume is resampled in **patient coordinates** onto the pixel grid of the slice on screen (nearest neighbour). So one label map fits every reconstruction of its scan, whatever the pixel size and field of view:

| Series (same scan, 11:18:49) | Pixel size |
|---|---|
| CT SSFreeze | 0.352 mm |
| CT CCTA DL | 0.377 mm |
| CT MEDIASTINUM DL | 0.732 mm |
| CT LUNGS 0.625MM | 0.744 mm |

**Same scan, not just same patient.** All CT series of TC_PTID share one frame of reference. The calcium-score scan is still a separate breath-hold about 4.6 mm away, 6 minutes earlier. A layer is therefore drawn only on slices from the acquisition it was made on: same `FrameOfReferenceUID` and `AcquisitionTime` within 2 s.
- **Heart layers:** CCTA, SSFreeze, Mediastinum and Lungs.
- **Epicardial fat:** the calcium score shows `eat_cac.nii.gz` (where the fat was measured); the CCTA-scan series show `eat_labels.nii.gz` (the same fat mapped onto the CCTA).
- **Other series:** the Scout (a projection) and the report images get nothing. The window says "made on another scan".

**Tests:** `tests/test_overlay.py` (6 tests):
- Labels land on the same millimetres at 0.377 and 0.744 mm pixels.
- The same-scan rule (SSFreeze's truncated time counts; the calcium score does not).
- Colours and legend, and the layer list.
- On the real data, the heart covers the same area (within 5 %) on a CCTA slice and the Lungs slice at the same height, and is refused on the calcium score, which gets its measured fat instead.

## MPR, HU readout and position sync in the DICOM tab (step 14)

Each window of the **🪟 DICOM Multi-View** tab has a **View** choice: *Axial*, *Coronal* or *Sagittal*. The sidebar has **🔗 Sync position (B–D ikut A)**. The display code is `scripts/slice_view.py`.

**Coronal and sagittal (MPR).**
- The series is read once as a volume and cached (SimpleITK, int16; about 120 MB for a 224-slice series).
- Planes are shown the radiological way: superior at the top and the patient's right on the viewer's left. Sagittal has anterior on the left. Orientation letters are at the edges.
- Planes keep their physical proportions: a CCTA coronal plane is 0.377 mm across and 0.625 mm between slices, so it is drawn 193 × 140 mm.
- Axial still reads single files, so non-volume folders (Scout, report images) keep working. MPR needs at least 10 slices.
- The segmentation overlay (step 13) works in all three views: labels are resampled onto a one-voxel-thick cut of the volume.

**HU readout.**
- Each window is a small HTML canvas (`streamlit.components.v1.html`) that carries the HU values of the plane. Plain HTML/JS, so no new dependencies.
- Hovering shows the HU, the structure under the cursor (with an overlay on), and the position in patient mm (LPS), e.g. `HU 509 · LV blood pool · x 24.6, y 12.7, z −78.1 mm (LPS)`.
- RGB report images have no HU, so they show the position only.

**Position sync.** With the toggle on, window A leads, and B–D follow by **position in the patient**, not slice number:
- **Same direction as A:** a window takes its plane nearest to A's, whatever its spacing. For example, CCTA slice 113 (z = −78.1 mm, 0.625 mm slices) gives calcium-score slice 29 (z = −79.0 mm, 2.5 mm slices). Its slider is locked while it follows. If A's position is outside the series, the window says so.
- **Another direction:** the window keeps its own slider and draws a dashed line where A's plane crosses it.
- The calcium-score scan is a separate breath-hold, so the same scanner position there is a few mm off anatomically.
- Each slider has its own key per series and view. Streamlit resets a widget whose label or range changes, which happens between series with different slice counts.

**Tests:** `tests/test_slice_view.py` (11 tests):
- A marker in the superior-left-anterior corner lands where each view should show it, and the readout's millimetres match SimpleITK's.
- Overlay labels line up with the image in all three views.
- Position matching and crossing lines.
- A localizer is not mistaken for an axial slice.
- Windowing and blending, and the canvas payload.
- On the real CCTA, an axial plane cut from the volume equals the slice read from its file, pixel for pixel.

## Mesh quality (CardioTwin XR roadmap, step 8, 9 Oct 2026)

`build_heart_model.py` now meshes the 8 heart structures together (`label_meshes`):
- **Shared surfaces.** Each structure's surface is where its smoothed indicator (σ 1 mm) beats every other one, so neighbours meet on one surface. Before, at the joins there were gaps of up to 5 mm and overlaps of 1–2 mm.
- **Joint smoothing.** All structures are smoothed as one connected mesh (`smooth_jointly` + `taubin`, 40 half-steps), so shared surfaces move together.
- **Volumes match the voxels** within about 1 % (myocardium −2.5 %).
- **Fixed bug:** `filter_taubin` was called with `nu=-0.53`. trimesh subtracts `nu`, so both half-steps shrank the mesh, and meshes were 4–22 % too small. trimesh's Laplacian also isn't normalised at zero-length edges, which is why the code has its own `taubin`.
- **Baked ambient occlusion** as `COLOR_0`, used by glTFast and three.js. Faces shared with a neighbour stay neutral.
- **See-through walls** (myocardium, alpha < 1) drop their shared faces in the GLB, to avoid z-fighting. The STL/OBJ exports keep closed, cleaned shells for printing.
- **New LV reference:** EDV 36.4, ESV 15.0 mL, EF 58.9 % (was 35.3 / 14.3 / 59.5).
- **Checks:** GLB 10.7 MB, 239k triangles, Khronos validator 0 issues; Unity 38/38 checks (LV 32.70 → 14.99 mL). Tests: `tests/test_meshing.py`, plus a closed-STL test in `test_pipeline.py`.
- **Known limitation:** the rule-based "myocardium" label is a thin sheet around the LA and in front of the aortic root, which shows as small pits and covers part of the aorta. The deep-learning chambers (step 5) would fix this.
- **Refreshed in step 10:** step 11's saved numbers (`phase_motion.py`) came from the old shrunken meshes. It was re-run on 10 Oct 2026 (see "Measured motion").

## Great vessels (CardioTwin XR roadmap, step 6, 9 Oct 2026)

`scripts/vessels.py` runs TotalSegmentator's open `total` task (no licence needed; ~75 s on the RTX 2060) and writes `vessels_dl.nii.gz` on the heart_labels grid. When that file exists, `build_heart_model.py` merges it into the model (`vessels.merge`):

| Model structure | From TotalSegmentator | May replace (rule labels) |
|---|---|---|
| Superior vena cava (8) | `superior_vena_cava` | background, catch-all myocardium, RA, PA, rule SVC |
| **Inferior vena cava (9, new)** | `inferior_vena_cava` | background, myocardium, RA |
| **Pulmonary veins (10, new)** | `pulmonary_vein` | background, myocardium |
| **Descending aorta (11, new)** | `aorta`, minus the part touching the LV | background, myocardium |
| LA blood pool (2) | `atrial_appendage_left` | background, myocardium |
| Aorta (3) | `aorta`, the part touching the LV | background, myocardium |

**What changed for TC_PTID:**
- **The measured chambers keep every voxel:** LV 32.7, RV 48.5 mL, and the EF curve is unchanged.
- **SVC:** the rule-based SVC lay in the right atrial appendage and now goes to the RA. TotalSegmentator's SVC (3.9 mL) is in the right place.
- **LA** 61.1 → 64.0 mL: its appendage is added.
- **Aorta** 23.7 → 29.5 mL: the thin "myocardium" sheet over the aortic root is gone, so the ascending aorta is visible.
- **New:** IVC 18.5 mL, pulmonary veins 11.5 mL, descending aorta 34.2 mL.
- **Myocardium** 99 → 87 mL: it had wrongly held parts of the vessels.

**Motion of the new structures:**
- **IVC:** rides with the RA, like the SVC.
- **Pulmonary veins:** ride with the LA.
- **Descending aorta:** distends only. It's fixed to the spine, so it doesn't follow the valve plane.

**Where the vessels end:** where the scan ends (z 0–140 mm), with flat caps. **The aortic arch is above the scan and is not added.**

**Checks:**
- 12 structures in the GLB (12.8 MB); Khronos validator 0 issues.
- Unity 46/46 checks, including "descending aorta behind the LA" and "IVC below the RA".
- All STLs watertight.
- Tests: `tests/test_vessels.py` (5).

**Known cosmetic issue:** banding along the descending aorta in the test renders. Wider or along-the-vessel smoothing didn't remove it (and wider smoothing pinched the pulmonary veins). It's likely normal faceting under the strong test light, so it's left to Phase 2's materials.

The DICOM-tab overlay (step 13) still shows the rule labels only.

## Coronary arteries (CardioTwin XR roadmap, step 7, 13 Oct 2026)

`scripts/coronaries.py` runs TotalSegmentator's `coronary_arteries` task (licensed like `heartchambers_highres`, free for non-commercial use; ~3.5 min on the RTX 2060) on CT CCTA DL:
```
.venv\Scripts\python scripts\coronaries.py              # after segment_heart.py
```
- **Output:** `output/3d_models/coronaries_dl.nii.gz`, the arteries on the CT's own grid (0.38 × 0.38 × 0.63 mm), cropped to the heart plus 10 mm, with specks under 2 mm³ removed; `coronaries_dl.json` (volume, pieces, versions).
- **TC_PTID:** 2.17 mL in 12 pieces. The task has one class for all coronary arteries (no branch names). The renders show the LAD down the anterior interventricular groove to the apex and branches in the AV grooves; small distal branches are missing or broken up.
- TotalSegmentator predicts them on a 0.7 mm grid and copies the result onto the CT's grid, so the mask has 0.7 mm steps.

**In the model** (`build_heart_model.py`, when the file exists):
- **Their own mesh** (`coronary_mesh`). They are 2–4 mm wide, too thin for the 1 mm shared-surface meshing, so they are meshed from the fine mask: smoothed by 0.35 mm (`COR_SIGMA_MM`; at 0.25 mm the 0.7 mm steps showed as beads), cut at level 0.4 (`COR_LEVEL`; keeps the thin branches in one piece), 30 Taubin half-steps. 52k triangles, 2.3 mL, 12 closed pieces. The light model meshes them at 2 voxels (12k triangles).
- **A label on the 1 mm grid**, "Coronary arteries" (13): the same smoothing resampled to 1 mm, on background voxels only, before the fat layer (2.2 mL). It is in `labels.raw` for the slice tool and the pointer, and the fat drops its faces that look onto the arteries, so they are not wrapped in tunnels.
- **Motion:** none of their own. Like the fat they count as the structure they lie on (`CARRIED`), so they ride on the beating surface: relative to the nearest heart surface they move 0.3 mm (median; at most 1.9 mm) at peak systole.
- **Colour:** orange-red in the labelled look; the realistic look gives them the oxygenated-vessel material.
- The other structures are unchanged; the model centre moved by 0.07 mm.

**Checks (13 Oct 2026):**
- Khronos validator 0 issues (347k triangles; light model 86k).
- Unity: import 66/66, case package 23/23, interface 29/29.
- The case-package line-up check uses half the inset (0.75 mm) for the coronaries, because 1.5 mm reaches past the centre of a 3 mm artery (94 % of vertices land in their label).
- Tests: `tests/test_coronaries.py` (3): a 3 mm artery is meshed where it lies at its volume and kept on the 1 mm grid; the arteries move with the wall they lie on; the cleanup and the crop.

## Epicardial fat layer (CardioTwin XR roadmap, step 9, 9 Oct 2026)

**The problem.** The EAT is measured on the 45 % calcium-score scan (73.4 mL). Mapped onto the 75 % CCTA, the mask:
- overlapped the heart by 13 mL, and
- lay mostly on non-fat tissue (only 26 of 70 mL is fat on the CCTA).

So the old fat mesh was patchy and partly inside the wall.

**The display layer (`fat_layer`).** The model now shows the CCTA's own fat that is:
- in the fat window (−190 to −30 HU, after 0.5 mm smoothing) and outside the heart;
- within 4 mm of the mapped EAT mask;
- in pieces that touch the heart (within 3 mm). Fat beyond the pericardium line isn't connected, so it's left out.

The fat becomes model label 12 and is meshed with the other structures (`label_meshes`), so it fills the AV and interventricular grooves without gaps or overlaps.

**Layered see-through rule (`hidden_faces`, `LAYER`).**
- The wall keeps the faces under the fat, so there's no hole in it.
- The fat, as the outer layer, drops the faces it shares with the wall.
- `faces_towards` also drops the fat's inner faces that look back at the heart within 3 mm. The 110 HU heart outline stops 1–2 mm short of the fat, so those faces aren't shared, and through the translucent fat they shaded as deep crevices.
- The fat gets half the ambient occlusion (`EAT_AO_SCALE`).

**Volumes.** Display layer 60.3 mL of voxels; the closed `heart_eat.stl` is 52.7 mL. The measured EAT stays 73.4 mL (`eat_stats.json`, GLB `extras.eat`). The display layer is for viewing, not a measurement.

**Checks:**
- GLB 15.1 MB, 336k triangles (the fat coating 77k); default face budget raised to 450k so every structure stays at marching-cubes step 1.
- Khronos validator 0 issues; Unity 46/46 checks.
- LV EDV 36.4 / ESV 15.0 mL, unchanged.
- Tests: `tests/test_fat_layer.py` (3).

**Left for Phase 2:** the fat is pale cream and smooth. The realistic yellow, lobulated look belongs to the materials work.

## Motion (CardioTwin XR roadmap, step 10, 10 Oct 2026)

**Four problems in the procedural motion** (`build_heart_model.py`):
1. **Shared surfaces came apart while beating.** Each structure moved by its own rule, so the surfaces meshed together in step 8 separated, by up to 9 mm (RV / septum, aorta / myocardium). In Unity this showed as dark gaps and crossing structures.
2. **The wall lost a fifth of its volume in systole** (−21 %), although myocardium is nearly incompressible. A fixed 22 mm "cavity radius" treated half of the wall as cavity; this LV's cavity is 14–20 mm.
3. **The atria and great vessels were dragged towards the apex** with the valve plane. Step 11's measurement suggested this: the model's whole-heart volume change was twice the measured one.
4. **The IVC moved like a ventricle wall** (6.5 mm). Its rule from step 6 sent it through the ventricular field.

**Changes:**
- **Blended motion (`motion_blend`, `MOTION_SIGMA_MM` 3 mm).** Every point moves by Σ w_k(p) × structure k's own motion, where w_k is the structure's smoothed indicator, normalised to sum 1.
  - The motion depends on the position only, so a vertex shared by two structures moves the same in both meshes.
  - Inside a structure (more than about 6 mm from a neighbour), it keeps its own motion.
  - The fat counts as the structure it lies on, so it moves with it.
- **Wall volume.** The LV cavity radius is fitted per height and angle (`fit_cavity`, 24 × 16 bins). Inside it the cavity scales as before, so the EF is unchanged. A 10 mm wall outside it grows in cross-section by 1/(1 − shortening), so it thickens as it shortens.
- **Anchors.** The valve-plane descent fades to zero 40 mm above the mitral plane (`ANCHOR_MM`: the veins hold the atrial roof) and, for the aorta and pulmonary artery, 70 mm above it (`ARTERY_ANCHOR_MM`: the arch and hila).
  - The atria therefore stretch and fill as the floor descends. The old separate "atrial filling" expansion is removed; with the stretch it counted the filling twice.
- **IVC** held by the diaphragm: it moves only where it blends into the RA.
- **RV:** its free-wall weighting used the farthest of the points evaluated together. It now uses a fixed RV radius, so the same point always moves the same.
- **LV curve.** It now uses the LV's own morph targets, as in the GLB, so the Python and Unity numbers agree exactly.

**Results (peak ventricular systole, closed shells):**

| | Before | After |
|---|---|---|
| Worst seam gap between structures | 7.6 mm (atrial systole 9.1) | 0.0 mm |
| Myocardium volume | −21 % | +1.4 % (LV wall −2 %) |
| Fat volume | +5.6 % | 0 % |
| LA / RA volume (filling) | +27 / +25 % | +16 / +16 % |
| Whole heart (no fat, IVC, descending aorta) | −5.2 % | −1.8 % |
| IVC mean motion | 6.5 mm | 0 (moves at the RA junction only) |
| LV EDV / ESV / EF | 36.4 / 15.0 mL / 58.9 % | 36.4 / 15.1 mL / 58.4 % |

- **Against the 45 % scan** (step 11, re-run): the whole-heart volume ratio is now 0.982 vs a measured 0.969. It was 0.932. The direction mismatch remains (see "Measured motion").
- **Atrial systole is unchanged** (LA −37 %, RA −28 %). It is large next to the LV's A-wave filling (+3.9 mL), but this LV (33 mL at rest) is small for an adult, probably a limit of the rule-based labels. The DL chambers (step 5) should settle it before tuning.

**Checks:**
- Unity 46/46 checks; LV in Unity 32.70 → 15.14 mL, equal to Python.
- Khronos validator 0 issues.
- Tests: `tests/test_motion.py` (5): cavity EF and wall volume, anchors, blending across a boundary, fat on its carrier, the RV field.

## Case package (CardioTwin XR roadmap, step 11, 10 Oct 2026)

`scripts/case_package.py` bundles one case for the XR app, so its Case panel and slice tool need nothing else:
```
.venv\Scripts\python scripts\case_package.py [--case-id DEMO-001] [--no-light]     # after build_heart_model.py; ~25 s with the light model
```

| File (`output/case_package/<case id>/`) | Contents |
|---|---|
| `heart.glb` | The animated model (`heart_animated.glb`). |
| `heart_light.glb` | The same model for standalone headsets: built from the same inputs with a 120k face budget (marching cubes at 2 voxels): 86k instead of 347k triangles (13 Oct 2026, with the coronaries), 4 MB, same structures and centre. Leave it out with `--no-light`. |
| `case.json` | The case card (schema `cardiotwin-case/1`). See below. |
| `ct.raw` | The 1 mm CCTA the model was built from, 193 × 193 × 140: uint16 little-endian, HU + 1024, x fastest, then y, then z (Unity's `Texture3D` order). 10.4 MB. |
| `labels.raw` | The model's labels on the same grid (uint8, same order), including the vessels and the fat layer. 5.2 MB. |

**`case.json`:**
- **Card fields:** `case_id`, `created`, and the notice "Research prototype. Simulated motion. Not for clinical use."
- **`structures`:** for each mesh, its name, label number, colour and volume. Also:
  - `kind`: *segmented*, *approximate*, or *display layer* (the fat);
  - `method`: how the shape was obtained;
  - `anchor_m`: a point well inside it, for callouts.
- **`measurements`:** the Case panel entries, each with a display `text` and a `status`:
  - epicardial fat 73.4 mL and −78.5 HU (*measured*, on the calcium-score scan);
  - calcium score 17.5 (*automatic, not reviewed*), plus the reviewed score once a reader saves a review in the app;
  - the radiology report's 0 (*reported*);
  - "CAD model: In development".
- **`motion`:** marked *simulated*, with the heart rate, clip and the procedural LV EDV / ESV / EF.
- **`grid`:** dims, spacing, origin and direction, with `voxel_from_gltf` and `voxel_from_unity`. These are 4 × 4, row-major, and map a model position in metres to a voxel index; the Unity one includes glTFast's X mirror.
  - The GLB now records the patient-space point at the model's origin (`extras.origin_lps_mm`), which ties the two together.
- **`ct`:** the HU offset and window presets (CCTA, soft tissue, fat).
- **No patient identifiers.** No names, IDs, UIDs or scan dates; the case ID is chosen with `--case-id`.

**Unity:**
- `unity/Assets/Heart/Scripts/HeartCase.cs` loads a package (e.g. from `Assets/StreamingAssets/Cases/<case id>/`). It gives:
  - the card as `[Serializable]` classes (`JsonUtility`);
  - the volumes as arrays or `Texture3D` (CT `R16`: HU = sample × 65535 − 1024; labels `R8`, point-filtered);
  - `LocalToVoxel` / `LocalToUvw` / `VoxelIndex` for points in the heart's local space.
- `Editor/CasePackageCheck.cs` (**Tools → Heart → Run case package check**, or `-executeMethod UTM.Heart.EditorTools.CasePackageCheck.Run`) checks a package against the imported model.

**Checks (10 Oct 2026):**
- **Unity: 22/22.** The card loads; the volumes match the grid; the package's GLB is the imported one; the textures are built.
  - For every structure, mesh vertices moved 1.5 mm inwards land in its own label: 100 % for the blood pools and vessels, 92–94 % for the thin wall, fat and pulmonary veins.
  - Contrast blood at the LV, LA and aorta anchors: 543, 433 and 479 HU. Fat at the fat anchor: −96 HU.
  - Python gives the same numbers, so glTFast's mirrored X is handled.
- **Tests:** `tests/test_case_package.py` (3).
  - The transforms agree with SimpleITK (glTF and Unity).
  - The raw layout is right.
  - A package built from the patient's outputs in a temporary folder lines up with its model and has no identifying keys or dates.

**Next (Phase 3):** the slice tool samples `Texture3D` at the cut plane through `LocalToUvw`, and the Case panel lists `measurements`.

## Unity / XR (step 12)

**The GLB is valid glTF 2.0.** The [Khronos glTF-Validator](https://github.com/KhronosGroup/glTF-Validator) reports 0 errors, 0 warnings, 0 infos and 0 hints on `heart_animated.glb`. Before 29 Sep 2026 it reported zero-length vertex normals on the myocardium and EAT meshes: marching cubes left near-zero-area triangles, which Unity shades black. `mask_to_mesh()` now drops faces without a normal. They have no area or volume, so the LV reference values are unchanged. `tests/test_pipeline.py::test_glb_normals_are_unit_length` guards this.

**Tested setup:** Unity **6000.6.3f1**, built-in render pipeline, [glTFast](https://docs.unity3d.com/Packages/com.unity.cloud.gltfast@6.20/manual/index.html) **6.20.0** (`com.unity.cloud.gltfast`). The test project is `C:\Users\MUBIN\UnityProjects\HeartXR` on the office PC. It is kept outside OneDrive because Unity's `Library/` cache is several GB.

**Import into a Unity project:**
1. Package Manager → *Add package by name* → `com.unity.cloud.gltfast`, version `6.20.0`.
2. Copy `unity/Assets/Heart/` from this project into the Unity project's `Assets/`. Copy `output/3d_models/heart_animated.glb` into `Assets/Heart/` too, and again after every pipeline run.
3. For the case package: copy `output/case_package/DEMO-001/` to `Assets/StreamingAssets/Cases/DEMO-001/`, then **Tools → Heart → Run case package check**.
4. **Tools → Heart → Run import check** (66 checks, report in the Console), then **Tools → Heart → Create demo scene**. Open `Assets/Heart/HeartDemo.unity` and press Play: the heart beats, and a small panel sets heart rate, wall opacity and visibility.

**What glTFast makes of it:**
- A root object with an **Animator but no Animator Controller**; glTFast's Editor import does not create one.
- Thirteen **SkinnedMeshRenderers** (nine without `vessels.py` and `coronaries.py`), each with the blend shapes `ventricular_systole`, `atrial_systole` and `great_artery_distension`. Weights keep glTF's 0–1 scale.
- A looping **`CardiacCycle`** clip (0.833 s at 72 bpm) with 36 blend-shape curves.
- Transparent `glTF/PbrMetallicRoughness` materials for the myocardium and EAT.
- **Coordinates:** 1 unit = 1 m, so the heart is 12 × 14 × 10 cm with the vessels. glTFast mirrors X: Unity −X = patient left, +Y = superior, +Z = anterior. A camera at +Z looking towards −Z sees the anterior view with the patient's left on the right.

**`HeartController`** (`unity/Assets/Heart/Scripts/HeartController.cs`) goes on the imported root. It plays the clip through a `PlayableGraph`, so no controller asset is needed, and offers the web viewer's controls:
- `heartRate` (speed = bpm / 72) and `playing`.
- `wallOpacity`, applied through a `MaterialPropertyBlock`, so the imported material is not changed.
- `SetVisible(name, on)`, `SetPhase(0..1)` (e.g. freeze at end-systole), and `Phase` / `PhaseName`.

**Verified in Unity:**
- The import check (`HeartImportCheck.cs`, see below) passes all **66** checks (13 Oct 2026), including the tissue shaders, materials, contact shadow and the mouse controls' manipulation.
- The interface check (`HeartUICheck.cs`) passes all **29** checks (see "Interface" and "Product polish").
- The LV blood pool deformed by Unity's blend shapes measures **52.32 → 22.77 mL** (rest → 0.325 R-R; deep-learning chambers), the same as the Python pipeline.
- The case package check (`CasePackageCheck.cs`) passes all **23** checks (see [Case package](#case-package-cardiotwin-xr-roadmap-step-11-10-oct-2026)).
- `output/visualizations/unity_preview.png` is Unity's own render of the demo scene (RTX 2060): the realistic look at rest, at mid-systole and with the see-through wall, the labelled look, and a close-up.

Run the checks from the command line:
```
"C:\Program Files\Unity\Hub\Editor\6000.6.3f1\Editor\Unity.exe" -batchmode -nographics -projectPath C:\Users\MUBIN\UnityProjects\HeartXR -executeMethod UTM.Heart.EditorTools.HeartImportCheck.Run -logFile check.log
"C:\Program Files\Unity\Hub\Editor\6000.6.3f1\Editor\Unity.exe" -batchmode -projectPath C:\Users\MUBIN\UnityProjects\HeartXR -executeMethod UTM.Heart.EditorTools.HeartImportCheck.RunRenderPreview -logFile render.log
```
The first exits with 0 when every check passes and writes `Logs/heart_import_check.txt`. The second renders `Logs/heart_preview_*.png`; leave out `-nographics` so the GPU renders.

### Realistic materials (CardioTwin XR roadmap, step 12, 10 Oct 2026)

`HeartMaterials` (`unity/Assets/Heart/Scripts/HeartMaterials.cs`) goes on the heart root next to `HeartController`. It switches between two looks with `CurrentLook`:
- **Realistic:** procedural tissue shaders (`unity/Assets/Heart/Shaders/`).
- **Labelled:** the GLB's own flat colours.

The demo scene uses the realistic look. **Tools → Heart → Create demo scene** creates the material assets in `Assets/Heart/Materials/` once. Later Inspector edits to them are kept.

| Structure | Material | Shader |
|---|---|---|
| Myocardium (LV wall) | Wall | `Heart/Muscle (see-through)`: opacity from `HeartController.wallOpacity` |
| Right ventricle | Ventricle muscle | `Heart/Muscle` |
| LA, right atrium | Atrial muscle (darker, more purple) | `Heart/Muscle` |
| LV blood pool | Blood | `Heart/Vessel` |
| Aorta, descending aorta, pulmonary veins | Oxygenated vessel (red) | `Heart/Vessel` |
| Pulmonary artery, SVC, IVC | Deoxygenated vessel (blue) | `Heart/Vessel` |
| Epicardial fat | Fat | `Heart/Fat` |

- **Colours follow the textbook convention for blood:** red = oxygenated, blue = deoxygenated. So the pulmonary artery is blue and the pulmonary veins are red.
- **Outer surfaces.** The rule labels have no separate RV or atrial wall, so the outer surfaces of the RV and atria are drawn as muscle.

**How the shaders work** (`HeartTissue.cginc`, built-in render pipeline surface shaders):
- **No texture coordinates needed.** Marching-cubes meshes have none, so all detail is 3D noise in millimetres.
  - **Muscle:** fibre striations on a helix around the long axis (−50°, 1.2 mm apart) plus mottling.
  - **Fat:** rounded lobules from cellular noise (5 mm), with soft grooves.
  - **Vessels:** faint mottling.
- **Bumps without tangents.** They use screen-space derivatives (Mikkelsen 2010).
- **Pattern stays on the tissue.** In play mode `HeartMaterials.StoreRestPositions()` copies each mesh with its rest positions in UV channel 3. The blend shapes move the vertices but not the UVs, so the pattern stays put as the heart beats. The imported meshes are not changed. In the Editor's scene view the pattern slides slightly instead.
- **Baked ambient occlusion** (vertex colours) darkens the creases.
- **Built-in "studio" reflection.** A soft sky and a soft box above front-left. Mixed-reality rooms give no reflection probe, so the shaders bring their own.
- **Translucency.** The fat is translucent (more opaque at grazing angles and in the grooves) and draws after the see-through wall (render queues 3000 / 3001). Both write depth first, so their own folds sort correctly.
- **Opacity slider.** `baseColorFactor` keeps glTFast's property name, so `HeartController`'s opacity works unchanged.

**Checks:**
- **Import check:** 55/55 at step 12. The 4 shaders compile with 0 errors; all 12 structures get their tissue material; the wall opacity keeps the muscle colour; the render order is right; the labelled look restores the GLB materials; rest positions are stored, with the blend shapes kept.
- **Previews:** `RunRenderPreview` now writes `heart_preview_{rest,systole,see_through,labelled,closeup}.png`.
- The demo scene has a fill and a rim light besides the key light.

**Notes:**
- **Colour space.** See step 13: Linear is recommended.
- **Render pipeline.** These are built-in render pipeline shaders. A standalone mixed-reality headset (e.g. Quest 3) would normally use URP, which needs a port. The noise and tissue code in `HeartTissue.cginc` carries over; the surface-shader wrappers don't.

### Lighting and contact shadow (CardioTwin XR roadmap, step 13, 12 Oct 2026)

**Contact shadow** (`HeartContactShadow.cs` on the heart root, shader `Heart/Contact Shadow`):
- A soft shadow on the desk just under the heart's lowest point (0.5 mm below it), so the heart looks like it sits there.
- It's built once from the meshes' rest shape on a 128 × 128 grid seen from above:
  - a dark core where the heart comes within 3 cm of the desk, blurred 4 mm;
  - a wide, faint footprint (35 %) under the whole heart, blurred 15 mm: the room light it blocks.
- It's drawn as black with transparency (85 %), so in mixed reality it darkens the passthrough video of the real table.
- No real-time shadow maps, so it costs next to nothing in a headset. In play mode it builds itself; tools call `Build()`.
- Since step 14 the shadow lies on the desk in the space of the heart's parent and is rebuilt for the heart's current pose.

**Room-style lighting** (the demo scene, **Tools → Heart → Create demo scene**):
- **Ambient:** a gradient, light above (sky), mid (equator) and darker towards the floor (ground), like a room.
- **Lights:** a warm key light from above the front (1.0), a cool fill from the side (0.35) and a rim light from behind (0.45). None cast shadow maps.
- **Desk:** a 1 m light-grey desk under the heart, named "Desk (turn off in mixed reality)".
- **Camera:** 20° above, like someone seated at the desk, against a dim room-grey background.

**Colour space (Project Settings → Player → Other Settings → Color Space):**
- The same scene was rendered in Gamma (the test project's setting) and in Linear.
- **Linear is recommended.** The tissue colours are richer, highlights look natural, the desk is a believable light grey, and the labelled (glTF) look is no longer washed out.
- Gamma over-brightens the desk and the labelled colours.
- It's a project-wide setting that Unity reimports for. Mixed-reality headsets (e.g. Quest) are normally set up in Linear too.
- **The test project was switched to Linear on 12 Oct 2026**, at the user's request.

**Checks:**
- **Import check:** 58/58. The contact-shadow shader compiles; the desk sits 0.5 mm under the heart's lowest point; the shadow is darkest 0.97, 0.35 under the middle of the heart, and 0 at its edges.
- **Play mode** (scratch project): the heart beats (LV weight 0 → 0.99), with rest-position meshes, tissue materials, the look switch and the contact shadow built at runtime.

### Interaction (CardioTwin XR roadmap, step 14, 12 Oct 2026)

**`HeartManipulator`** (on the heart root) rotates, zooms and moves the heart about its own centre, smoothly, and resets the view.
- **It reads no input itself.** The mouse drives it now (`HeartMouseInput`); hand tracking can call the same `Orbit` / `Rotate` / `Zoom` / `Move` / `ResetView` later.
- **Space.** Poses are in the space of the heart's parent (the "stage": the desk, or a mixed-reality anchor), with +Y up from the desk.
- **Resting on the desk** (`restOnDesk`, on by default). After every change the heart is lowered or raised so its lowest point sits 0.5 mm above the desk, and the contact shadow is rebuilt for the new pose:
  - while it moves, the lowest point comes from 8,000 sample vertices, and the shadow is rebuilt at most 5 times a second;
  - once it stops, every vertex is used, so the heart is exact to the vertex.
- **Zoom** goes from ×0.5 to ×4, and the shadow's wide footprint fades as the heart is lifted.

**`HeartMouseInput`** (desktop; Input Manager):

| Action | Mouse / key |
|---|---|
| Rotate: sideways about the vertical, up/down about the camera's horizontal axis | left-drag |
| Zoom | scroll wheel |
| Move across the desk | right- or middle-drag |
| Reset view | R, or double-click |

- Presses that start on the control panel are left to the panel (`HeartController.PanelContains`), and the panel shows these controls.

**Demo scene:** the heart has `HeartManipulator` and `HeartMouseInput`. Press Play and drag.

**Checks:**
- **Import check: 62/62.**
  - Rotating 90° turns the heart about its centre (0.000 mm drift), with the lowest point 0.50 mm above the desk.
  - Tilting 60° and zooming 2× keeps it resting on the desk, with the shadow rebuilt flat on the desk.
  - Moving goes 5 cm across the desk; the upward part is ignored.
  - Zoom stops at ×4; reset restores the original pose exactly.
- **Play mode** (scratch project): a turn and zoom sent through the manipulator settles on the desk (0.50 mm) with the shadow on the desk; no errors from our code.
- **Preview:** the render now includes `heart_preview_turned.png`.

### Interface: toolbar, Layers, slice tool, Case panel, labels (CardioTwin XR roadmap, steps 15–20, 12 Oct 2026)

`HeartUI` (on the heart root) builds the interface at runtime with Unity's uGUI. It needs the package `com.unity.ugui` (2.6.0, bundled with Unity 6), which was added to the test project's `Packages/manifest.json`. It replaces `HeartController`'s debug panel.

| Part | What it does |
|---|---|
| **Toolbar** (bottom) | **Rotate / Zoom / Slice** (what a left-drag does), **Play / Pause**, heart rate **− / +** (5 bpm), **Reset view** |
| **Layers** (left) | **Heart** and **EAT overlay** (display layer) on / off, **CT on the cut**, **Realistic look** (else labelled), **Wall opacity**, one toggle per structure, **Reset view** |
| **Case** (right) | The case package's card: epicardial fat 73.4 mL and −78.5 HU (*measured*), calcium score 17.5 (*automatic, not reviewed*) and the report's 0, "CAD model: In development", the simulated motion, and the notice |
| **Slice** (bottom right, while slicing) | **Axial / Coronal / Sagittal**, **Flip**, the plane's position, and a **2D view of the CT** in the plane of the cut |
| **Label** | The structure under the pointer, its volume, and "display layer" for the fat, in a box joined by a line to the point on it |

**Slice tool (`HeartSlice`, shader `Heart/Slice Cap`):**
- **The plane follows the heart's own axes** (axial, coronal, sagittal: the CT's axes), so the cut stays anatomical however the heart is turned. Drag with Slice selected to move it.
- **Cutting:** every tissue shader discards what lies on the far side of the plane (`HeartClip.cginc`, a global `_HeartClipPlane`). That includes the depth pass of the see-through wall and fat, and the labelled look, which now uses the same shaders with the GLB's colours and no detail.
- **The cut face shows the patient's CT** from the case package (3D textures), masked to the heart's cross-section and tinted by structure. Or, with **CT on the cut** off, the structures' colours.
- **Alignment:** the CT and the meshes come from the same grid, so they line up. The interface check finds the LV at the same voxel from both sides (0.0000 voxel apart), with 543 HU there (contrast blood).
- **The 2D CT view** shows the whole scan in the plane of the cut, in radiological orientation (the patient's right on the left; axial: anterior up).

**Labels (`HeartPicker`):**
- **Mesh colliders on the rest shape** follow the heart's pose. Hidden structures and the cut-away part are skipped.
- **On the cut face** the structure comes from the case package's labels.
- The heart's motion is ignored for picking, so in systole the label can be a few millimetres off.

**Mouse:** presses on the interface are left to it (EventSystem). `HeartMouseInput.mode` is set by the toolbar.

**Headset later:** `HeartUI.renderMode = WorldSpace` puts the panels 0.8 m in front of the viewer, facing them. The interface draws after all the tissue (render queue 4000), so panels are never painted over by the see-through wall or fat.

**Checks:**
- **Interface check: 20/20** (`HeartUICheck.Run`, or **Tools → Heart → Run interface check**):
  - each toolbar button;
  - the Layers toggles, the slider and the look;
  - the Case panel text;
  - the axial and coronal cuts through the LV (plane, CT alignment, 2D CT view 193 × 193 and 193 × 140);
  - labels on the cut face and on the surface (and skipping a hidden structure);
  - world-space mode.
- **Import check: 63/63** (adds the slice-face shader).
- **Play mode** (scratch project): the interface builds itself; Slice → Coronal cuts with the CT face and view; Slice again restores the heart; 0 errors.
- **Pictures:** `HeartUICheck.RunPreview` writes `Logs/heart_ui_{overview,axial_cut,coronal_cut}.png`.

### Product polish (CardioTwin XR roadmap, Phase 4, steps 21–24, 12 Oct 2026)

**21. The CardioTwin XR look** (`HeartUI`, `HeartIcons`, shader `Heart/UI Glass`):
- **Title bar:** "CardioTwin XR" with a heart icon, the case picker, **Seated / Standing**, **Desktop / Headset**.
- **Notice pill:** "Simulated motion · Research prototype".
- **Toolbar icons:** rotate, zoom, slice, play / pause, reset, recentre. They're drawn at runtime from simple shapes, so there are no image assets.
- **Frosted glass on the desktop:** the panels blur and tint the scene behind them, with one screen grab per frame shared by all panels. In the headset they're plain translucent, because grabbing the screen costs too much there.

**22. Performance for headsets:**
- **Light model.** The case package's `heart_light.glb` has 86k triangles instead of 347k (13 Oct 2026, with the coronaries).
- **Headset quality** (`HeartStage.quality`):
  - it loads the light model;
  - the panels float in world space without blur;
  - the wall starts opaque (less transparent overdraw; the opacity slider still works).
- **Measured in play mode on the office PC** (RTX 2060, Quest 3-like stereo frame, 4128 × 2208, 4× MSAA, model, panels and contact shadow):

| Set-up | Triangles | Median | 90th percentile |
|---|---|---|---|
| Desktop quality, full model | 336,242 | 10.3 ms | 11.2 ms |
| Headset quality, light model | 86,236 | 8.3 ms | 11.9 ms |

- **PC VR** (your target: Quest Link / SteamVR on this PC). 8–10 ms for both eyes fits 72–90 fps, but with little headroom at 90 fps.
- **Standalone Quest.** Its mobile GPU is several times slower, so it would need the light model plus further savings: fewer transparent layers, lower MSAA, URP.
- **These are estimates.** Nothing was measured in a headset, since none is connected yet.

**23. Comfort** (`HeartStage` on the heart's parent, "Heart Stage", which also holds the desk):
- **Recentre** (toolbar button or **C**). In the headset it puts the stage in front of the viewer: the heart 55 cm ahead with its front facing them, the desk top 40 cm (seated) or 55 cm (standing) below the eyes. On the desktop the camera does not move, so it puts the stage back where it started.
- **Seated / Standing** sets those heights.
- **Panels that follow:** in the headset they sit 1 m wide, 0.85 m ahead and a little below the eyes, upright. When the viewer has turned more than 30° away they glide back in front and stop. Small head movements don't move them.

**24. Case picker:**
- **Choosing a case.** The case button in the title bar lists the case packages in `Assets/StreamingAssets/Cases/`. Choosing one loads it at runtime (`HeartStage.LoadCase`):
  - its model (the light one in Headset quality), with glTFast and legacy animation, the only kind that works in builds; `HeartController` plays legacy clips too;
  - its CT and case card;
  - the same components as the demo scene's heart, resting on the same desk.
  It then replaces the current heart.
- **Adding a case:** run `python scripts/case_package.py --case-id <ID>` and copy `output/case_package/<ID>/` to `Assets/StreamingAssets/Cases/`.
- **Switching quality** reloads the current case with the matching model.

**Checks:**
- **Interface check: 29/29.** The 20 from Phase 3, plus:
  - the title and notice;
  - toolbar icons;
  - glass on the desktop and none in world space;
  - the case list;
  - Seated / Standing;
  - Recentre seated (40.0 cm below the eyes, 55.0 cm ahead, facing) and standing (55.0 cm), and back home on the desktop;
  - world-space panels 0.85 m ahead, still after a 15° turn and back in front (0.5°) after 60°.
- **Import check: 64/64** (adds the glass shader).
- **Play mode** (scratch project, with a second test package): loading a case at runtime gives the full model (336,242 triangles). It beats on its legacy clip (LV weight 0 → 0.99), with tissue materials, CT, case card and title, contact shadow, resting on the desk (0.50 mm), and one heart in the scene. Headset quality reloads the light model (86,236 triangles), with world-space panels, no glass and an opaque wall. 0 errors.

**Not done yet (PC VR):**
- The XR rig: OpenXR, XR Interaction Toolkit, grab/scale of the heart, and a world-space panel instead of the desktop one.
- No OpenXR runtime (Meta Quest Link or SteamVR) is installed on the office PC, so nothing can be tested in a headset yet.
- The model is about 347k triangles (with the coronaries) with two transparent meshes. That should be fine for PC VR; a standalone headset would need a lighter version (Phase 4).

## Limitations (V1)
- The rule-based right heart (RA, RV, pulmonary artery, SVC) is approximate. In this coronary-timed scan the right-heart blood (~130 HU) is close to myocardium (~150 HU), so it is split by anatomy rules, not by density. With the TotalSegmentator licence the model uses the deep-learning chambers instead (step 5); they look better but are not validated against expert labels.
- The coronary arteries (step 7) show the main branches only; small distal branches are missing or broken up.
- The LV cavity is the contrast blood pool only, so trabeculae and papillary muscles are excluded (33 mL at 75 % phase).
- The rules are tuned for this scanner and protocol (GE Revolution, SmartPrep CCTA). Check `segmentation_overlay.png` for every new patient.
- The motion is synthetic (literature-typical values), not measured from this patient. The one other phase available (45 %, non-contrast) suggests the model overstates the global deformation of the outer surface; see [Measured motion](#measured-motion-step-11).
- EAT: the model was trained on 18 patients from two scanners (Philips, Siemens) and has not been validated on GE scans such as TC_PTID. Treat the patient's EAT as a research estimate and check the QC figure. It is not a clinical measurement.
