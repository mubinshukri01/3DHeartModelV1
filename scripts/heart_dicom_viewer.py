"""
==============================================================
  Heart DICOM 4-Window Viewer — Side-by-Side Comparison
  UTM-Claude Research Project | rohayanti@utm.my

  Cara guna:
    pip install streamlit pydicom SimpleITK numpy matplotlib
    streamlit run heart_dicom_viewer.py
==============================================================
"""

import streamlit as st
import pydicom
import numpy as np
import os, glob, warnings
from pathlib import Path
import heart_io as hio
import overlay as ov
import slice_view as sv
import streamlit.components.v1 as components
from heart_3d_tab import render_3d_tab
warnings.filterwarnings("ignore")

# ── Decoder check ─────────────────────────────────────────
def _check_sitk():
    try:
        import SimpleITK; return True
    except ImportError:
        return False

SITK_OK = _check_sitk()
if SITK_OK:
    import SimpleITK as sitk

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Heart DICOM 4-View",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background:#f0f2f6; }
  div[data-testid="stSidebarContent"] { background:#fff; }
  .win-box {
    background:#fff; border-radius:10px; padding:12px;
    box-shadow:0 2px 8px rgba(0,0,0,.10); margin-bottom:8px;
  }
  .win-title {
    font-size:13px; font-weight:700; color:#c0392b;
    text-transform:uppercase; letter-spacing:.5px; margin-bottom:6px;
  }
  .info-card {
    background:#fff; border-left:4px solid #c0392b;
    border-radius:6px; padding:10px 14px; margin:4px 0;
    box-shadow:0 1px 4px rgba(0,0,0,.07);
  }
  .info-label { font-size:10px; font-weight:700; color:#888; text-transform:uppercase; }
  .info-value { font-size:13px; font-weight:600; color:#1a1a2e; }
  h1,h2,h3,h4 { color:#1a1a2e !important; }
  .stSlider label { font-size:12px !important; }
  .block-container { padding-top:1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Window presets ────────────────────────────────────────
PRESETS = {
    "Cardiac (W:350 L:40)":     (350,  40),
    "Bone/CAC (W:1500 L:300)":  (1500, 300),
    "Lung (W:1500 L:-600)":     (1500, -600),
    "Mediastinum (W:350 L:50)": (350,  50),
}
CMAPS = ["gray", "bone", "hot", "inferno", "plasma", "viridis"]

# ── Helpers ───────────────────────────────────────────────
def safe_get(ds, tag, default="—"):
    try:
        v = getattr(ds, tag, None)
        return str(v) if v is not None else default
    except Exception:
        return default

@st.cache_data(show_spinner=False)
def load_series(folder):
    # "**" with recursive=True also matches the folder itself, so this one
    # pattern already covers top-level files (adding "*.dcm" loaded them twice)
    files = sorted(glob.glob(os.path.join(folder, "**", "*.dcm"), recursive=True))
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True, stop_before_pixels=False)
            slices.append((f, ds))
        except Exception:
            continue
    slices.sort(key=lambda x: (
        int(getattr(x[1], "InstanceNumber", 9999))
        if hasattr(x[1], "InstanceNumber") else
        float(getattr(x[1], "SliceLocation", 0))
    ))
    return slices

def detect_series(base):
    result = {}
    p = Path(base)
    if list(p.glob("*.dcm")):
        result[p.name] = str(p)
    for sub in sorted(p.iterdir()):
        if sub.is_dir() and list(sub.glob("*.dcm")):
            result[sub.name] = str(sub)
    return result

def read_pixel(filepath, ds):
    """Read pixel array — SimpleITK first (decodes the JPEG Lossless files), then pydicom.

    Returns (array, error, SimpleITK image or None); the image carries the slice's position in the
    patient, which the segmentation overlay needs.
    """
    if SITK_OK:
        try:
            img = sitk.ReadImage(str(filepath))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            return (arr[0] if arr.ndim == 3 else arr), None, img
        except Exception as e:
            pass
    try:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        return arr * slope + intercept, None, None
    except Exception as e:
        return None, str(e), None


@st.cache_data(show_spinner=False)
def slice_heights(folder):
    """Height (patient z, mm) of each slice of load_series(folder), in the same order."""
    return [float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else float("nan")
            for _, ds in load_series(folder)]


@st.cache_resource(show_spinner="Memuatkan volume untuk MPR ...", max_entries=4)
def load_volume(folder):
    """Series as one int16 volume for coronal / sagittal views; None if the folder is not a volume."""
    try:
        vol = hio.load_series(folder)
    except Exception:
        return None
    if vol.GetSize()[2] < sv.MIN_MPR_SLICES:
        return None
    return sitk.Cast(sitk.Clamp(vol, sitk.sitkFloat32, -32768, 32767), sitk.sitkInt16)


def is_axial(ds):
    try:
        iop = [float(v) for v in ds.ImageOrientationPatient]
        return abs(iop[0]) > 0.9 and abs(iop[4]) > 0.9
    except (AttributeError, ValueError, IndexError):
        return False


def overlay_labels(overlay, ds, pl):
    """Chosen segmentation layer on a plane: (labels or None, palette or None, HTML legend line or None)."""
    if not overlay:
        return None, None, None
    name, entries = overlay
    entry = ov.layer_for(ds, entries)
    if entry is None or pl.ref is None:
        return None, None, f"🧩 {name}: tiada untuk siri ini (dibuat pada imbasan lain / made on another scan)"
    path, source, palette = entry
    lab = sv.labels_on(pl, ov.label_slice(ov.load_labels(path), pl.ref))
    sr, sc = pl.spacing
    chips = " · ".join(
        f'<span style="display:inline-block;width:9px;height:9px;border-radius:2px;'
        f'background:rgb({r * 255:.0f},{g * 255:.0f},{b * 255:.0f})"></span> {label} {area / 100:.1f} cm²'
        for label, (r, g, b), area in ov.present(lab, palette, sr * sc))
    return lab, palette, f"🧩 {name} <span style='color:#888'>(dari {source})</span>: " + (chips or "tiada label pada slice ini")


def slice_slider(win_id, series, view, n, follow):
    """Slice slider of one window, series and view; when following window A it is set and locked.

    Streamlit gives a widget a new identity (and resets it) when its label, range or help change, so
    each series and view has its own key and the other arguments stay fixed for that key.
    """
    key = f"slice_{win_id}_{view}_{series}"
    if follow is not None:
        st.session_state[key] = follow + 1
    elif not 1 <= st.session_state.get(key, 0) <= n:
        st.session_state[key] = n // 2 + 1
    name = "Slice" if view == "Axial" else view
    return st.slider(f"{name} (1–{n})", 1, n, key=key, disabled=follow is not None) - 1


def render_window(win_id, col, overlay=None, overlay_alpha=0.45, sync=None, height=430):
    """Render one viewer window inside a column.

    overlay: (layer name, entries from ov.layers()) or None. sync: dict shared by the windows of one run
    when position sync is on (window A, rendered first, leads), else None.
    """
    label = ["A", "B", "C", "D"][win_id]
    with col:
        st.markdown(f'<div class="win-title">🪟 Window {label}</div>', unsafe_allow_html=True)

        folder = st.text_input(
            f"📁 Path folder",
            key=f"path_{win_id}",
            placeholder=hio.DATASET,
            label_visibility="collapsed",
        )
        if not folder or not os.path.isdir(folder):
            st.caption("Masukkan path folder DICOM")
            return

        series = detect_series(folder)
        if not series:
            st.warning("Tiada .dcm dijumpai")
            return

        sel = st.selectbox(
            "Siri", list(series.keys()),
            key=f"series_{win_id}",
            label_visibility="collapsed",
        )
        slices = load_series(series[sel])
        if not slices:
            st.warning("Tiada slice")
            return

        view = st.radio("View", list(sv.VIEWS), horizontal=True, key=f"view_{win_id}",
                        label_visibility="collapsed")
        preset = st.selectbox(
            "Preset", list(PRESETS.keys()),
            key=f"preset_{win_id}",
            label_visibility="collapsed",
        )
        ww, wl = PRESETS[preset]
        cmap = st.selectbox(
            "Colormap", CMAPS,
            key=f"cmap_{win_id}",
            label_visibility="collapsed",
        )

        leader = sync.get("leader") if sync is not None and win_id > 0 else None
        ds = slices[0][1]
        follow = None
        if view == "Axial":
            n = len(slices)
            if leader and leader["axis"] == 2 and is_axial(ds):
                heights = slice_heights(series[sel])
                gap = float(np.nanmedian(np.abs(np.diff(heights)))) if n > 1 else 1.0
                follow = sv.nearest_z(heights, leader["coord"], max(gap, 0.5))
            idx = slice_slider(win_id, sel, view, n, follow)
            fpath, ds = slices[idx]
            arr, err, img = read_pixel(fpath, ds)
            if arr is None:
                st.error("❌ Tidak dapat baca pixel.")
                if not SITK_OK:
                    st.code("pip install SimpleITK", language="bash")
                return
            pl = sv.from_slice(arr, img)
            where = f"#{idx + 1}/{n}"
        else:
            vol = load_volume(series[sel]) if SITK_OK else None
            if vol is None:
                st.warning(f"{view}: perlukan siri volume (≥ {sv.MIN_MPR_SLICES} slice) / needs a volume series")
                return
            axis = sv.VIEWS[view]
            n = vol.GetSize()[axis]
            if leader and leader["axis"] == axis:
                follow = sv.nearest(vol, view, leader["coord"])
            idx = slice_slider(win_id, sel, view, n, follow)
            pl = sv.plane(vol, view, idx)
            where = f"{view} {idx + 1}/{n}"

        lab, palette, legend = overlay_labels(overlay, ds, pl)
        rgb = sv.compose(pl.hu, ww, wl, cmap, lab, palette, overlay_alpha)
        if sync is not None and win_id == 0:
            sync["leader"] = {"axis": pl.axis, "coord": pl.coord}
        line = sv.crossing(pl, leader["axis"], leader["coord"]) if leader else None
        rng = f"HU {int(pl.hu.min())} to {int(pl.hu.max())}" if pl.hu.ndim == 2 else "RGB image"
        names = {k: v[0] for k, v in palette.items()} if palette else None
        components.html(sv.html(rgb, pl, height, title=f"{sel[:22]}  {where}", footer=f"{rng} · hover: HU",
                                orient=sv.ORIENTATION[view], labels=lab, names=names, line=line),
                        height=height)

        # Compact metadata strip
        sr, sc = pl.spacing
        if view == "Axial":
            sl_loc  = safe_get(ds, "SliceLocation")
            thick   = safe_get(ds, "SliceThickness")
            st.caption(f"Loc: {sl_loc} mm · Thick: {thick} mm · {ds.Rows if hasattr(ds,'Rows') else '?'}×{ds.Columns if hasattr(ds,'Columns') else '?'}")
        else:
            st.caption(f"{view} at {'xyz'[pl.axis]} = {pl.coord:.1f} mm · pixel {sr:.3f} × {sc:.3f} mm · {pl.hu.shape[0]}×{pl.hu.shape[1]}")
        if leader and follow is None and leader["axis"] == pl.axis:
            st.caption("🔗 Kedudukan window A di luar siri ini / window A's position is outside this series")
        if legend:
            st.markdown(f'<div style="font-size:11px;line-height:1.5">{legend}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫀 Heart DICOM\n**4-Window Viewer**")
    st.caption("UTM-Claude Research Project\nrohayanti@utm.my")
    st.divider()

    # Decoder status
    if SITK_OK:
        st.success("SimpleITK ✅ — compressed DICOM supported")
    else:
        st.error("SimpleITK ❌")
        st.code("pip install SimpleITK", language="bash")
        st.caption("Restart Streamlit selepas install")

    st.divider()

    # Layout selector
    layout_mode = st.radio(
        "Layout",
        ["2×2 (4 windows)", "1×2 (2 windows)", "1×1 (1 window)"],
        index=0,
    )

    st.divider()

    # Global colormap override
    global_cmap = st.selectbox("🎨 Sync Colormap (semua window)", ["(individual)"] + CMAPS)
    global_preset = st.selectbox("🪟 Sync Preset (semua window)", ["(individual)"] + list(PRESETS.keys()))

    st.divider()

    # Segmentation overlay (all windows): pipeline labels, only on the scan they were made on
    overlay_layers = ov.layers()
    overlay_name = st.selectbox(
        "🧩 Segmentation overlay", ["(none)"] + list(overlay_layers),
        help="Label jantung / lemak dari pipeline, dilukis pada slice. Hanya pada rekonstruksi imbasan "
             "yang sama (CCTA, Lungs, Mediastinum, SSFreeze untuk jantung; lemak juga pada CALCIUM SCORE).")
    overlay_alpha = st.slider("Overlay opacity", 0.1, 1.0, 0.45, 0.05, disabled=overlay_name == "(none)")
    overlay = None if overlay_name == "(none)" else (overlay_name, overlay_layers[overlay_name])

    sync_on = st.toggle(
        "🔗 Sync position (B–D ikut A)", value=False,
        help="Window B–D ikut kedudukan window A dalam badan pesakit (mm), walaupun jarak slice berbeza. "
             "Satah arah lain menunjukkan garis putus-putus di tempat satah A melintas. Imbasan CALCIUM SCORE "
             "ialah tahanan nafas lain, jadi kedudukan yang sama di situ lari beberapa mm secara anatomi.")

    st.divider()
    st.caption("Dataset: TC_PTID")
    st.caption("Owner: utm4dt@gmail.com")
    st.caption("Scanner: GE Revolution CT")

# ═══════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("## 🫀 Heart DICOM — Multi-Window Viewer & 3D Model")
st.caption("Masukkan path folder berbeza pada setiap window untuk perbandingan CT series secara serentak.")

tab_dicom, tab_3d = st.tabs(["🪟 DICOM Multi-View", "🫀 3D Heart Model"])

with tab_3d:
    render_3d_tab()

with tab_dicom:
    # ── Collapsible sections ──────────────────────────────────
    with st.expander("📋 Panduan Langkah & Siri Dataset", expanded=False):
        cl1, cl2 = st.columns(2)
        with cl1:
            st.markdown("**🚀 Cara Guna**")
            st.markdown("""
1. Masukkan path ke folder DICOM dalam kotak setiap window
2. Pilih CT series dari dropdown
3. Gunakan slider untuk scroll slices
4. Tukar Window Preset untuk kontras berbeza
5. Bandingkan berbilang series secara serentak
        """)
        with cl2:
            st.markdown("**📂 Siri Dataset (TC_PTID)**")
            series_info = {
                "CT CALCIUM SCORE":   ("🦴", "50 slices",  "2.5mm, CAC scoring"),
                "CT CCTA DL":         ("🫀", "224 slices", "0.625mm, Deep Learning recon"),
                "CT LUNGS 0.625MM":   ("🫁", "224 slices", "High-res lung CT"),
                "CT MEDIASTINUM DL":  ("🫀", "224 slices", "Mediastinal + DL"),
                "CT PatientReport":   ("📋", "Reports",    "Laporan klinikal"),
                "CT Scout":           ("📡", "2 slices",   "Scout/Topogram"),
                "CT SSFreeze":        ("❄️",  "224 slices", "Motion-corrected cardiac"),
                "SR Dose Record":     ("📊", "1 file",     "Rekod dos DICOM SR"),
                "SR FUJI HL7":        ("🏥", "1 file",     "HL7 radiologi report"),
            }
            for name, (icon, count, desc) in series_info.items():
                st.markdown(f"**{icon} {name}** — {count} · _{desc}_")

    with st.expander("📄 CT Patient Report — Maklumat Klinikal", expanded=False):
        pr1, pr2, pr3 = st.columns(3)
        with pr1:
            st.markdown("""
**🧑‍⚕️ Maklumat Pesakit**
| | |
|---|---|
| Patient ID | e56c515cd66cb79d |
| Study Date | 22 Mei 2023 |
| Modality | CT |
| Scanner | GE Revolution CT |
| KVP | 100–120 kV |
        """)
        with pr2:
            st.markdown("""
**🔬 Parameter CT**
| | |
|---|---|
| Protocol | 5.19 Coronary SmartPrep-DLIR |
| Reconstruction | Deep Learning (DLIR) |
| Slice Thickness | 0.625 – 2.5 mm |
| Matrix | 512 × 512 |
| X-Ray Current | 301–552 mA |
        """)
        with pr3:
            st.markdown("""
**🫀 Prosedur Kardiovaskular**
| | |
|---|---|
| Jenis CT | CT Coronary Angiography (CCTA) |
| CAC Score | Coronary Artery Calcium |
| Kontras | SmartPrep gating |
| Organ | Jantung, aorta, mediastinum |
| Tujuan | Penilaian risiko kardiovaskular |
        """)
        st.info(f"📁 Fail laporan pesakit tersimpan di: `{os.path.join(hio.DATASET, 'CT PatientReport')}`")

    st.divider()

    # ═══════════════════════════════════════════════════════════
    #  VIEWER WINDOWS
    # ═══════════════════════════════════════════════════════════

    # Apply global overrides to session state
    if global_cmap != "(individual)":
        for i in range(4):
            st.session_state[f"cmap_{i}"] = global_cmap
    if global_preset != "(individual)":
        for i in range(4):
            st.session_state[f"preset_{i}"] = global_preset

    sync = {} if sync_on else None          # window A (rendered first) writes its position here
    height = {"2×2 (4 windows)": 430, "1×2 (2 windows)": 520}.get(layout_mode, 660)

    if layout_mode == "2×2 (4 windows)":
        row1 = st.columns(2, gap="small")
        row2 = st.columns(2, gap="small")
        with st.container():
            render_window(0, row1[0], overlay, overlay_alpha, sync, height)
            render_window(1, row1[1], overlay, overlay_alpha, sync, height)
        with st.container():
            render_window(2, row2[0], overlay, overlay_alpha, sync, height)
            render_window(3, row2[1], overlay, overlay_alpha, sync, height)

    elif layout_mode == "1×2 (2 windows)":
        cols = st.columns(2, gap="small")
        render_window(0, cols[0], overlay, overlay_alpha, sync, height)
        render_window(1, cols[1], overlay, overlay_alpha, sync, height)

    else:
        cols = st.columns([1, 2, 1])
        render_window(0, cols[1], overlay, overlay_alpha, sync, height)

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("🫀 Heart DICOM 4-Window Viewer · UTM-Claude Research Project · rohayanti@utm.my")
