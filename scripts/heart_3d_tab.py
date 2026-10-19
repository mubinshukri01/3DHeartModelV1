"""
Streamlit "3D Heart Model" tab - used by heart_dicom_viewer.py
UTM-Claude Research Project | rohayanti@utm.my
"""
import json, os
import streamlit as st
import streamlit.components.v1 as components

import heart_io as hio

GLB = os.path.join(hio.OUT_MODELS, "heart_animated.glb")
GLTF = os.path.join(hio.OUT_MODELS, "heart_animated.gltf")
STATS = os.path.join(hio.OUT_MODELS, "segmentation_stats.json")
ANIM = os.path.join(hio.OUT_MODELS, "animation_info.json")
OVERLAY = os.path.join(hio.OUT_VIS, "segmentation_overlay.png")
CYCLE = os.path.join(hio.OUT_VIS, "cardiac_cycle.png")
EAT_STATS = os.path.join(hio.OUT_MODELS, "eat_stats.json")
EAT_QC = os.path.join(hio.OUT_VIS, "eat_overlay.png")
CA_STATS = os.path.join(hio.OUT_MODELS, "calcium_stats.json")
CA_QC = os.path.join(hio.OUT_VIS, "calcium_overlay.png")


def _load_json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _viewer_html(glb_path, mtime, bpm):
    from build_heart_model import viewer_html_inline
    return viewer_html_inline(glb_path, bpm)


def _run_eat(log):
    """Step 1b (epicardial fat), when the calcium-score series and the trained model are available."""
    import eat_patient, eat_segment
    if not os.path.isdir(os.path.join(hio.DATASET, eat_patient.EAT_SERIES)):
        log(f"EAT skipped: no `{eat_patient.EAT_SERIES}` series in the dataset")
    elif not os.path.exists(eat_segment.MODEL_PATH):
        log("EAT skipped: no trained model (run `python eat_benchmark.py --save-model`)")
    else:
        eat_patient.main(log=log)
        import calcium_score                                   # step 1c needs the EAT of step 1b
        calcium_score.main(log=log)


def _run_pipeline(series, bpm):
    import segment_heart, build_heart_model, case_package
    with st.status("Running 3D heart pipeline ...", expanded=True) as s:
        segment_heart.main(series, log=st.write)
        _run_eat(st.write)
        build_heart_model.main(bpm, log=st.write)
        case_package.main(log=st.write)                     # keeps the XR case package in step with the model
        s.update(label="Pipeline complete ✅", state="complete")
    _viewer_html.clear()


def render_3d_tab():
    has_data = os.path.isdir(hio.DATASET)
    series_dirs = sorted(d for d in os.listdir(hio.DATASET)
                         if os.path.isdir(os.path.join(hio.DATASET, d)) and d.startswith("CT"))         if has_data else []
    bpm = 72
    if series_dirs:  # local PC with DICOM: allow re-running the pipeline
        c1, c2, c3 = st.columns([3, 1.2, 1])
        with c1:
            default = series_dirs.index(hio.DEFAULT_SERIES) if hio.DEFAULT_SERIES in series_dirs else 0
            series = st.selectbox("Contrast CT series for segmentation", series_dirs, index=default,
                                  help="CT CCTA DL (contrast, 0.625 mm) gives the best result.")
        with c2:
            bpm = st.number_input("Heart rate (bpm)", 40, 150, 72, 1)
        with c3:
            st.write("")
            run = st.button("▶ Run pipeline", use_container_width=True, type="primary")
        if run:
            _run_pipeline(series, bpm)
    elif not os.path.exists(GLB):  # deployed server: outputs only, no DICOM
        st.warning(f"No 3D model found in `{hio.OUT_MODELS}` - upload heart_animated.glb and the JSON files.")
        return

    if not os.path.exists(GLB):
        st.info("No 3D model yet. Click **▶ Run pipeline** (≈10 s), or run\n\n"
                "`python segment_heart.py` then `python build_heart_model.py` in the scripts folder.")
        return

    stats, anim = _load_json(STATS) or {}, _load_json(ANIM) or {}
    # EAT is shown only when the current model actually contains the fat mesh
    eat = _load_json(EAT_STATS) if "Epicardial fat (EAT)" in anim.get("faces", {}) else None
    view, side = st.columns([3, 1.1])
    with view:
        components.html(_viewer_html(GLB, os.path.getmtime(GLB), int(anim.get("bpm", bpm))), height=620)
        st.caption("Drag to rotate · scroll to zoom · toggle structures & heart rate in the panel. "
                   "Rest pose = CT phase 75% R-R; motion is procedural (single-phase dataset)."
                   + (f" Pale cream = epicardial fat, measured on {eat['series']} "
                      f"({eat['cardiac_phase_percent']:.0f}% R-R) and mapped onto the 75% model." if eat else ""))
    with side:
        st.markdown("**📊 Segmentation (mL)**")
        for k, v in stats.get("labels", {}).items():
            st.markdown(f'<div class="info-card"><div class="info-label">{k}</div>'
                        f'<div class="info-value">{v} mL</div></div>', unsafe_allow_html=True)
        if stats:
            st.markdown(f'<div class="info-card"><div class="info-label">Whole heart</div>'
                        f'<div class="info-value">{stats["whole_heart_ml"]} mL</div></div>',
                        unsafe_allow_html=True)
        if eat:
            st.markdown("**🟡 Epicardial fat (EAT)**")
            st.markdown(f'<div class="info-card"><div class="info-label">EAT volume</div>'
                        f'<div class="info-value">{eat["eat_volume_ml"]} mL</div></div>'
                        f'<div class="info-card"><div class="info-label">EAT attenuation</div>'
                        f'<div class="info-value">{eat["eat_mean_hu"]} HU</div></div>', unsafe_allow_html=True)
            m = eat.get("model", {})
            st.caption(f"Measured on {eat['series']}, fat {eat['fat_range_hu'][0]} to {eat['fat_range_hu'][1]} HU. "
                       f"Method Dice {m.get('benchmark_dice_mean', float('nan')):.2f} vs expert masks "
                       f"({m.get('trained_on_patients', '?')}-patient Rodriguez benchmark). Research use only.")
        if anim:
            st.markdown("**🫀 Animation**")
            st.caption(f"{anim['bpm']:.0f} bpm · period {anim['period_s']:.2f} s · "
                       f"model LV EF {anim['lv_ef_percent']:.0f}% · GLB {anim['glb_mb']:.1f} MB")
        st.markdown("**⬇ Downloads**")
        with open(GLB, "rb") as fh:
            st.download_button("heart_animated.glb (web / Unity)", fh.read(),
                               "heart_animated.glb", "model/gltf-binary", use_container_width=True)
        for fn in ("heart_lv.stl", "heart_myocardium.stl") + (("heart_eat.stl",) if eat else ()):
            p = os.path.join(hio.OUT_MODELS, fn)
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    st.download_button(fn, fh.read(), fn, use_container_width=True)
        st.caption(f"All files: `{hio.OUT_MODELS}`")

    e1, e2 = st.columns(2)
    with e1:
        if os.path.exists(OVERLAY):
            with st.expander("🧩 Segmentation overlay (QC)", expanded=False):
                st.image(OVERLAY, use_container_width=True)
    with e2:
        if os.path.exists(CYCLE):
            with st.expander("📈 Cardiac cycle: LV volume & morph weights", expanded=False):
                st.image(CYCLE, use_container_width=True)
    if eat and os.path.exists(EAT_QC):
        with st.expander("🟡 Epicardial fat (QC): measured on the calcium score, mapped onto the CCTA", expanded=False):
            st.image(EAT_QC, use_container_width=True)
    _render_calcium()


def _render_calcium():
    """Agatston score: automatic suggestions, reader review (saved to calcium_review.json), report value."""
    s = _load_json(CA_STATS)
    if not s:
        return
    import calcium_score as cs
    from datetime import datetime
    lesions, auto, review = s["lesions"], s["automatic"], _load_json(cs.review_path())
    st.markdown("#### 🦴 Coronary calcium (Agatston)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Automatic (unreviewed)", f"{auto['agatston']:g}",
              help=f"{auto['lesions']} suggested lesion(s) in the coronary territory · CAC-DRS {auto['cac_drs']}")
    if review is not None:
        a, v, n = cs.score(lesions, set(review["coronary"]))
        c2.metric("Reviewed", f"{a:g}", help=f"{n} lesion(s) confirmed, {v:g} mm³ · reviewed {review['saved']}")
        c2.caption(f"CAC-DRS {cs.cac_drs(a)}")
    else:
        c2.metric("Reviewed", "—", help="Tick the coronary lesions below and save")
    rep = s.get("report_agatston")
    c3.metric("Radiology report", "—" if rep is None else f"{rep:g}")
    if not lesions:
        st.success("No calcium lesion in the coronary territory: Agatston 0.")
    else:
        with st.form("calcium_review"):
            st.caption("Each suggestion is a spot ≥ 130 HU (red) next to epicardial fat (yellow). Keep the ticks on "
                       "coronary calcium; untick noise, valve or aortic calcium, then save.")
            cols = st.columns(min(len(lesions), 5)); picks = {}
            for i, l in enumerate(lesions):
                with cols[i % len(cols)]:
                    img = os.path.join(cs.lesions_dir(), f"lesion_{l['id']}.png")
                    if os.path.exists(img):
                        st.image(img, use_container_width=True)
                    picks[l["id"]] = st.checkbox(f"#{l['id']}: {l['peak_hu']} HU, {l['area_mm2']:g} mm², score {l['agatston']:g}",
                                                 value=(l["id"] in review["coronary"]) if review else True, key=f"ca_{l['id']}")
            if st.form_submit_button("Save review"):
                with open(cs.review_path(), "w") as fh:
                    json.dump({"coronary": [i for i, v in picks.items() if v], "not_coronary": [i for i, v in picks.items() if not v],
                               "saved": datetime.now().isoformat(timespec="minutes")}, fh, indent=2)
                st.rerun()
    if os.path.exists(CA_QC):
        with st.expander("🦴 Calcium QC: every spot ≥ 130 HU near the heart", expanded=False):
            st.image(CA_QC, use_container_width=True)
            st.caption(f"Excluded automatically: {s['excluded']}. Agatston scaled by slice {s['slice_mm']:g} mm / 3 mm "
                       f"(× {s['agatston_scale']:g}).")
