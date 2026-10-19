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
import matplotlib.pyplot as plt
from pathlib import Path
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
def apply_window(arr, ww, wl):
    lo, hi = wl - ww/2, wl + ww/2
    return ((np.clip(arr, lo, hi) - lo) / max(hi-lo, 1) * 255).astype(np.uint8)

def safe_get(ds, tag, default="—"):
    try:
        v = getattr(ds, tag, None)
        return str(v) if v is not None else default
    except Exception:
        return default

@st.cache_data(show_spinner=False)
def load_series(folder):
    files = sorted(
        glob.glob(os.path.join(folder, "*.dcm")) +
        glob.glob(os.path.join(folder, "**/*.dcm"), recursive=True)
    )
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
    """Read pixel array — SimpleITK first (handles JPEG2000), then pydicom."""
    if SITK_OK:
        try:
            img = sitk.ReadImage(str(filepath))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            return (arr[0] if arr.ndim == 3 else arr), None
        except Exception as e:
            pass
    try:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        return arr * slope + intercept, None
    except Exception as e:
        return None, str(e)

def render_window(win_id, col):
    """Render one viewer window inside a column."""
    label = ["A", "B", "C", "D"][win_id]
    with col:
        st.markdown(f'<div class="win-title">🪟 Window {label}</div>', unsafe_allow_html=True)

        folder = st.text_input(
            f"📁 Path folder",
            key=f"path_{win_id}",
            placeholder=r"C:\3DHeartModelV1\dataset",
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

        n = len(slices)
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

        idx = st.slider(f"Slice (1–{n})", 1, n, n//2, key=f"slice_{win_id}") - 1
        fpath, ds = slices[idx]
        arr, err = read_pixel(fpath, ds)

        if arr is None:
            st.error("❌ Tidak dapat baca pixel.")
            if not SITK_OK:
                st.code("pip install SimpleITK", language="bash")
            return

        windowed = apply_window(arr, ww, wl)
        fig, ax = plt.subplots(figsize=(4, 4), facecolor="#111")
        ax.imshow(windowed, cmap=cmap, aspect="equal")
        ax.axis("off")
        hu_min, hu_max = int(arr.min()), int(arr.max())
        ax.text(4, 10, f"{sel[:18]}  #{idx+1}/{n}",
                color="white", fontsize=7, va="top",
                bbox=dict(facecolor="#c0392b", alpha=.85, boxstyle="round,pad=.2"))
        ax.text(4, arr.shape[0]-6, f"HU {hu_min}–{hu_max}",
                color="white", fontsize=7, va="bottom",
                bbox=dict(facecolor="#000", alpha=.6, boxstyle="round,pad=.2"))
        fig.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Compact metadata strip
        sl_loc  = safe_get(ds, "SliceLocation")
        thick   = safe_get(ds, "SliceThickness")
        st.caption(f"Loc: {sl_loc} mm · Thick: {thick} mm · {ds.Rows if hasattr(ds,'Rows') else '?'}×{ds.Columns if hasattr(ds,'Columns') else '?'}")

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫀 Heart DICOM\n**4-Window Viewer**")
    st.caption("UTM-Claude Research Project\nrohayanti@utm.my")
    st.divider()

    # Decoder status
    if SITK_OK:
        st.success("SimpleITK ✅ — JPEG2000 supported")
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
    st.caption("Dataset: TC_PTID")
    st.caption("Owner: utm4dt@gmail.com")
    st.caption("Scanner: GE Revolution CT")

# ═══════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("## 🫀 Heart DICOM — Multi-Window Viewer")
st.caption("Masukkan path folder berbeza pada setiap window untuk perbandingan CT series secara serentak.")

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
            "CT CALCIUM SCORE":   ("🦴", "100 slices", "2.5mm, CAC scoring"),
            "CT CCTA DL":         ("🫀", "448 slices", "0.625mm, Deep Learning recon"),
            "CT LUNGS 0.625MM":   ("🫁", "~300 slices","High-res lung CT"),
            "CT MEDIASTINUM DL":  ("🫀", "~300 slices","Mediastinal + DL"),
            "CT PatientReport":   ("📋", "Reports",    "Laporan klinikal"),
            "CT Scout":           ("📡", "2-4 slices", "Scout/Topogram"),
            "CT SSFreeze":        ("❄️",  "~300 slices","Motion-corrected cardiac"),
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
    st.info("📁 Fail laporan pesakit tersimpan di: `C:\\3DHeartModelV1\\dataset\\CT PatientReport\\`")

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

if layout_mode == "2×2 (4 windows)":
    row1 = st.columns(2, gap="small")
    row2 = st.columns(2, gap="small")
    with st.container():
        render_window(0, row1[0])
        render_window(1, row1[1])
    with st.container():
        render_window(2, row2[0])
        render_window(3, row2[1])

elif layout_mode == "1×2 (2 windows)":
    cols = st.columns(2, gap="small")
    render_window(0, cols[0])
    render_window(1, cols[1])

else:
    cols = st.columns([1, 2, 1])
    render_window(0, cols[1])

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("🫀 Heart DICOM 4-Window Viewer · UTM-Claude Research Project · rohayanti@utm.my")
