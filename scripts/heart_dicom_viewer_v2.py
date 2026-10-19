"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  3D Heart Viewer V2 — Rodriguez Benchmark Dataset (20 Patients)            ║
║  UTM Research | rohayanti@utm.my                                            ║
║                                                                             ║
║  Dataset structure:                                                         ║
║    C:\3DHeartModelV1\dataset\Rodriguez\                                     ║
║      ├── DICOM\   {PatientID}_{Slice}_input.dcm   ← all patients flat      ║
║      ├── IMAGES\  (PNG previews)                                            ║
║      └── MASKS\   {PatientID}_{Slice}_gt.dcm      ← segmentation masks     ║
║                                                                             ║
║  Features:                                                                  ║
║    • Auto-group 20 patients from flat DICOM folder by PatientID prefix      ║
║    • Tab 1 — 2D Multi-Planar (Axial · Coronal · Sagittal + strip)          ║
║    • Tab 2 — 3D Volume Rendering (Plotly interactive)                       ║
║    • Tab 3 — 3D Surface Mesh (Marching Cubes → Plotly Mesh3D)              ║
║    • Tab 4 — Mask Overlay (if MASKS folder present)                         ║
║                                                                             ║
║  Install:  pip install SimpleITK pydicom scikit-image plotly scipy streamlit ║
║  Run:      streamlit run heart_dicom_viewer_v2.py                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, re, glob
import numpy as np
import streamlit as st
from pathlib import Path
from collections import defaultdict

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="🫀 3D Heart Viewer | UTM",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "UTM Research — Rodriguez Dataset | rohayanti@utm.my"},
)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from scipy import ndimage as _ndi_check  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1117 !important; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.stTabs [data-baseweb="tab-list"]  { gap:4px; background:#161b22; border-radius:8px; padding:4px; }
.stTabs [data-baseweb="tab"]       { border-radius:6px; padding:8px 18px; font-weight:600; font-size:14px; }
.stTabs [aria-selected="true"]     { background:#1f6feb !important; color:white !important; }
.hdr {
    background: linear-gradient(135deg,#0d419d,#1f6feb);
    border-radius:10px; padding:14px 20px; margin-bottom:12px;
    color:white; font-size:18px; font-weight:700;
}
.info-box {
    background:#0d419d22; border:1px solid #1f6feb55; border-radius:8px;
    padding:10px 14px; margin:8px 0; font-size:13px;
}
div[data-testid="metric-container"] {
    background:#161b22; border-radius:8px; padding:10px; border:1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET PARSING — flat folder, naming: {PatientID}_{Slice}_input.dcm
# ══════════════════════════════════════════════════════════════════════════════

def extract_patient_id(filename: str) -> tuple[str, int]:
    """
    From filename like 'AEdu_033_input.dcm'
    returns ('AEdu', 33)
    Falls back to ('unknown', 0) if pattern doesn't match.
    """
    stem = Path(filename).stem                      # 'AEdu_033_input'
    # Remove trailing _input / _gt / _mask etc.
    stem = re.sub(r'_(input|gt|mask|label|seg)$', '', stem, flags=re.IGNORECASE)
    parts = stem.split('_')
    if len(parts) >= 2:
        pid   = parts[0]
        try:
            slnum = int(parts[1])
        except ValueError:
            slnum = 0
        return pid, slnum
    return stem, 0


@st.cache_data(show_spinner=False)
def scan_rodriguez_dataset(root_dir: str) -> dict:
    """
    Scan C:\\...\\Rodriguez\\ for:
        DICOM\   → {PatientID}_{n}_input.dcm
        MASKS\   → {PatientID}_{n}_gt.dcm  (optional)
        IMAGES\  → ignored (PNG previews)

    Returns:
        {
          'PatientID': {
              'dicom_files':  [(slice_num, filepath), ...],   # sorted
              'mask_files':   [(slice_num, filepath), ...],   # sorted, may be []
          },
          ...
        }
    """
    patients = defaultdict(lambda: {'dicom_files': [], 'mask_files': []})

    dicom_dir = os.path.join(root_dir, 'DICOM')
    masks_dir = os.path.join(root_dir, 'MASKS')

    # ── Scan DICOM folder ──────────────────────────────────────────────────
    dcm_dir_used = dicom_dir if os.path.isdir(dicom_dir) else root_dir

    dcm_patterns = ['*.dcm', '*.DCM', '*.ima', '*.IMA']
    dcm_candidates = []
    for pat in dcm_patterns:
        dcm_candidates.extend(glob.glob(os.path.join(dcm_dir_used, pat)))

    for fpath in dcm_candidates:
        fname = os.path.basename(fpath)
        if re.search(r'(gt|mask|label|seg)', fname, re.IGNORECASE):
            continue                              # skip masks in DICOM folder
        pid, slnum = extract_patient_id(fname)
        patients[pid]['dicom_files'].append((slnum, fpath))

    # ── Scan MASKS folder ──────────────────────────────────────────────────
    if os.path.isdir(masks_dir):
        mask_candidates = []
        for pat in dcm_patterns:
            mask_candidates.extend(glob.glob(os.path.join(masks_dir, pat)))

        for fpath in mask_candidates:
            fname = os.path.basename(fpath)
            pid, slnum = extract_patient_id(fname)
            if pid in patients:
                patients[pid]['mask_files'].append((slnum, fpath))

    # Sort slices and remove empty groups
    result = {}
    for pid, data in patients.items():
        dicom = sorted(data['dicom_files'], key=lambda x: x[0])
        masks = sorted(data['mask_files'],  key=lambda x: x[0])
        if dicom:
            result[pid] = {
                'dicom_files': dicom,
                'mask_files':  masks,
                'n_slices':    len(dicom),
                'has_masks':   len(masks) > 0,
            }

    return dict(sorted(result.items()))


# ══════════════════════════════════════════════════════════════════════════════
#  VOLUME LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_volume_from_files(filepaths_tuple: tuple) -> tuple:
    """
    Load a stack of 2D DICOM slices into a 3D volume.
    filepaths_tuple: tuple of (slice_num, filepath) pairs — sorted by slice_num.
    Returns (volume [Z,Y,X] float32, spacing_mm (x,y,z)).
    """
    filepaths = [fp for _, fp in filepaths_tuple]

    # ── Try SimpleITK first ────────────────────────────────────────────────
    if HAS_SITK:
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(filepaths)
            reader.MetaDataDictionaryArrayUpdateOn()
            image = reader.Execute()
            vol   = sitk.GetArrayFromImage(image).astype(np.float32)
            sp    = image.GetSpacing()            # (x_mm, y_mm, z_mm)
            return vol, sp
        except Exception:
            pass

    # ── Fallback: pydicom slice-by-slice ──────────────────────────────────
    if HAS_PYDICOM:
        arrays = []
        spacing_set = False
        sx = sy = sz = 1.0
        for fpath in filepaths:
            try:
                ds  = pydicom.dcmread(fpath, force=True)
                arr = ds.pixel_array.astype(np.float32)
                try:
                    arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                except Exception:
                    pass
                arrays.append(arr)
                if not spacing_set:
                    try:
                        sx = float(ds.PixelSpacing[0])
                        sy = float(ds.PixelSpacing[1])
                    except Exception:
                        pass
                    try:
                        sz = float(ds.SliceThickness)
                    except Exception:
                        pass
                    spacing_set = True
            except Exception:
                continue
        if arrays:
            return np.stack(arrays, axis=0), (sx, sy, sz)

    return None, (1.0, 1.0, 1.0)


@st.cache_data(show_spinner=False)
def load_mask_volume(filepaths_tuple: tuple):
    """Load binary mask slices → 3D bool volume."""
    filepaths = [fp for _, fp in filepaths_tuple]
    arrays = []

    if HAS_SITK:
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(filepaths)
            image = reader.Execute()
            return (sitk.GetArrayFromImage(image) > 0).astype(np.uint8)
        except Exception:
            pass

    if HAS_PYDICOM:
        for fpath in filepaths:
            try:
                ds  = pydicom.dcmread(fpath, force=True)
                arr = (ds.pixel_array > 0).astype(np.uint8)
                arrays.append(arr)
            except Exception:
                continue
        if arrays:
            return np.stack(arrays, axis=0)

    return None


@st.cache_data(show_spinner=False)
def get_dicom_meta(fpath: str) -> dict:
    """Read metadata from a single DICOM file."""
    if not HAS_PYDICOM:
        return {}
    try:
        ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
        return {
            'Patient ID':        str(getattr(ds, 'PatientID',        'N/A')),
            'Patient Name':      str(getattr(ds, 'PatientName',      'N/A')),
            'Patient Sex':       str(getattr(ds, 'PatientSex',       'N/A')),
            'Patient Age':       str(getattr(ds, 'PatientAge',       'N/A')),
            'Study Date':        str(getattr(ds, 'StudyDate',        'N/A')),
            'Modality':          str(getattr(ds, 'Modality',         'N/A')),
            'Manufacturer':      str(getattr(ds, 'Manufacturer',     'N/A')),
            'Study Description': str(getattr(ds, 'StudyDescription', 'N/A')),
            'Slice Thickness':   str(getattr(ds, 'SliceThickness',   'N/A')),
            'KVP':               str(getattr(ds, 'KVP',              'N/A')),
            'Rows':              str(getattr(ds, 'Rows',             'N/A')),
            'Columns':           str(getattr(ds, 'Columns',          'N/A')),
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def apply_window(vol: np.ndarray, wc: float, ww: float) -> np.ndarray:
    lo  = wc - ww / 2.0
    hi  = wc + ww / 2.0
    out = np.clip(vol, lo, hi)
    return ((out - lo) / (hi - lo + 1e-9)).astype(np.float32)


CT_PRESETS = {
    "Cardiac  (WC=50,  WW=350)":   (50,   350),
    "Soft Tissue (WC=40, WW=400)": (40,   400),
    "Chest (WC=-600, WW=1500)":    (-600, 1500),
    "Lung  (WC=-500, WW=1400)":    (-500, 1400),
    "Bone  (WC=400,  WW=1800)":    (400,  1800),
    "Custom":                       None,
}

PLOTLY_CMAP = {
    'gray': 'Gray', 'bone': 'Greys', 'hot': 'Hot',
    'plasma': 'Plasma', 'viridis': 'Viridis',
    'RdBu': 'RdBu', 'turbo': 'Turbo', 'jet': 'Jet',
}


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🫀 3D Heart Viewer V2")
    st.markdown("**Rodriguez Benchmark Dataset**  \n`rohayanti@utm.my | UTM`")
    st.markdown("---")

    # Library status
    def badge(ok, label):
        c = "#2ea043" if ok else "#f85149"
        i = "✅" if ok else "❌"
        return f'<span style="color:{c};font-size:12px">{i} {label}</span>'

    st.markdown(
        badge(HAS_SITK,    "SimpleITK") + "&nbsp;&nbsp;" +
        badge(HAS_PYDICOM, "pydicom")   + "<br>" +
        badge(HAS_PLOTLY,  "Plotly")    + "&nbsp;&nbsp;" +
        badge(HAS_SKIMAGE, "scikit-image"),
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Dataset root
    st.markdown("### 📁 Rodriguez Dataset")
    st.markdown(
        '<div class="info-box">Struktur folder:<br>'
        '<code>Rodriguez/DICOM/{ID}_{n}_input.dcm</code><br>'
        '<code>Rodriguez/MASKS/{ID}_{n}_gt.dcm</code></div>',
        unsafe_allow_html=True,
    )

    dataset_root = st.text_input(
        "Root Folder",
        value=r"C:\3DHeartModelV1\dataset\Rodriguez",
    )

    col_s, col_c = st.columns(2)
    if col_s.button("🔄 Scan", use_container_width=True):
        st.cache_data.clear()
    if col_c.button("🗑 Clear", use_container_width=True):
        st.cache_data.clear()

    patients = scan_rodriguez_dataset(dataset_root)

    if patients:
        n_pat = len(patients)
        st.success(f"✅ {n_pat} pesakit ditemui")
        selected_pid = st.selectbox(
            "👤 Pilih Pesakit",
            list(patients.keys()),
            format_func=lambda x: f"🫀 {x}  ({patients[x]['n_slices']} slices)",
        )
        pdata = patients[selected_pid]
        if pdata['has_masks']:
            st.info("🩺 Mask segmentasi tersedia")
    else:
        st.warning("⚠️ Tiada fail .dcm ditemui.\nSemak path folder dan kandungan DICOM\\.")
        selected_pid = None
        pdata = None

    st.markdown("---")

    # CT Window
    st.markdown("### 🎛️ CT Window / Level")
    preset_name = st.selectbox("Preset", list(CT_PRESETS.keys()))
    if preset_name == "Custom":
        wc = st.slider("Window Center (HU)", -1000, 3000, 50)
        ww = st.slider("Window Width  (HU)", 1,    4000, 350)
    else:
        wc, ww = CT_PRESETS[preset_name]
        st.info(f"WC = {wc} HU  |  WW = {ww} HU")

    colormap = st.selectbox(
        "Colormap (2D)",
        ['gray', 'bone', 'hot', 'plasma', 'viridis', 'RdBu', 'turbo', 'jet'],
    )

    st.markdown("---")
    st.markdown("### 🔬 3D Settings")
    iso_thresh    = st.slider("Isosurface HU", -300, 1000, 150)
    mesh_opacity  = st.slider("Mesh opacity",  0.10, 1.00, 0.70, 0.05)
    ds_step       = st.slider("Downsample step", 1, 8, 2,
                               help="Lebih besar = lebih laju, kurang detail")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="hdr">🫀 3D Heart Viewer V2 — Rodriguez Benchmark Dataset &nbsp;|&nbsp; '
    'UTM Research &nbsp;|&nbsp; rohayanti@utm.my</div>',
    unsafe_allow_html=True,
)

if not patients or not selected_pid:
    st.info("👈 Tetapkan **Root Folder** di sidebar dan tekan **Scan**.")
    with st.expander("📦 Install packages"):
        st.code("pip install SimpleITK pydicom scikit-image plotly scipy streamlit numpy matplotlib")
    st.stop()

# ── Load volume ────────────────────────────────────────────────────────────────
with st.spinner(f"⏳ Memuatkan {selected_pid} ({pdata['n_slices']} slices)…"):
    vol, spacing = load_volume_from_files(tuple(pdata['dicom_files']))

if vol is None:
    st.error("❌ Tidak dapat memuatkan DICOM. Pastikan pydicom atau SimpleITK dipasang.")
    st.stop()

nz, ny, nx = vol.shape
sx, sy, sz  = spacing

# ── Load mask (optional) ───────────────────────────────────────────────────────
mask_vol = None
if pdata['has_masks']:
    with st.spinner("Memuatkan mask segmentasi…"):
        mask_vol = load_mask_volume(tuple(pdata['mask_files']))

# ── Patient header ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patient ID",  selected_pid)
c2.metric("Slices (Z)",  nz)
c3.metric("Image size",  f"{ny} × {nx}")
c4.metric("Has Masks",   "✅ Yes" if pdata['has_masks'] else "—")

meta = get_dicom_meta(pdata['dicom_files'][0][1])
with st.expander("📋 DICOM Metadata", expanded=False):
    if meta:
        cols = st.columns(4)
        for i, (k, v) in enumerate(meta.items()):
            cols[i % 4].markdown(f"**{k}:** {v}")
    hu_lo, hu_hi = int(vol.min()), int(vol.max())
    st.markdown(
        f"**HU range:** {hu_lo} – {hu_hi} &nbsp;|&nbsp; "
        f"**Spacing:** {sx:.3f} × {sy:.3f} × {sz:.3f} mm"
    )

st.markdown("---")

# Apply windowing
vol_w = apply_window(vol, wc, ww)

# ── All patient summary strip (sidebar area — compact) ─────────────────────────
with st.expander(f"🗂️ Semua Pesakit ({len(patients)} patients)", expanded=False):
    pat_cols = st.columns(5)
    for i, (pid, pd) in enumerate(patients.items()):
        mask_icon = "🩺" if pd['has_masks'] else "  "
        is_sel    = "**" if pid == selected_pid else ""
        pat_cols[i % 5].markdown(
            f"{mask_icon} {is_sel}{pid}{is_sel}  \n"
            f"<small>{pd['n_slices']} slices</small>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_labels = ["📐 2D Multi-Planar", "🌐 3D Volume", "🫀 3D Surface Mesh"]
if pdata['has_masks']:
    tab_labels.append("🩺 Mask Overlay")

tabs = st.tabs(tab_labels)
tab_2d  = tabs[0]
tab_vol = tabs[1]
tab_msh = tabs[2]
tab_msk = tabs[3] if pdata['has_masks'] else None


# ════════════════════════════════════════════════════════
# TAB 1 — 2D MULTI-PLANAR
# ════════════════════════════════════════════════════════
with tab_2d:
    st.markdown(f"#### 📐 Multi-Planar Reconstruction — {selected_pid}")

    r1c1, r1c2, r1c3 = st.columns(3)
    z_idx = r1c1.slider("Axial    (Z)", 0, nz - 1, nz // 2, key="slz")
    y_idx = r1c2.slider("Coronal  (Y)", 0, ny - 1, ny // 2, key="sly")
    x_idx = r1c3.slider("Sagittal (X)", 0, nx - 1, nx // 2, key="slx")

    show_mask_2d = False
    if pdata['has_masks'] and mask_vol is not None:
        show_mask_2d = st.checkbox("Overlay segmentation mask", value=True, key="mask2d")

    # Main 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor='#0d1117')
    fig.suptitle(
        f"Patient: {selected_pid}  |  WC={wc}  WW={ww}  |  {colormap}",
        color='#c9d1d9', fontsize=11, fontweight='bold',
    )

    views = [
        (vol_w[z_idx, :, :],   f"Axial    Z = {z_idx}",  "←X→", "←Y→"),
        (vol_w[:, y_idx, :],   f"Coronal  Y = {y_idx}",  "←X→", "←Z→"),
        (vol_w[:, :, x_idx],   f"Sagittal X = {x_idx}",  "←Y→", "←Z→"),
    ]
    mask_views = []
    if show_mask_2d and mask_vol is not None:
        mask_views = [
            mask_vol[z_idx, :, :],
            mask_vol[:, y_idx, :],
            mask_vol[:, :, x_idx],
        ]

    for idx_v, (ax, (img, title, xl, yl)) in enumerate(zip(axes, views)):
        ax.imshow(img, cmap=colormap, aspect='equal', origin='lower', interpolation='bilinear')
        if mask_views:
            msk = mask_views[idx_v].astype(float)
            msk_alpha = np.where(msk > 0, 0.45, 0.0)
            overlay = np.zeros((*msk.shape, 4))
            overlay[..., 0] = 1.0     # red channel
            overlay[..., 3] = msk_alpha
            ax.imshow(overlay, aspect='equal', origin='lower')
        ax.set_title(title, color='#79c0ff', fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel(xl, color='#8b949e', fontsize=8)
        ax.set_ylabel(yl, color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e', labelsize=7)
        ax.set_facecolor('#0d1117')
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')

    # Crosshairs
    axes[0].axhline(y_idx, color='#f85149', alpha=0.5, lw=0.9, ls='--')
    axes[0].axvline(x_idx, color='#79c0ff', alpha=0.5, lw=0.9, ls='--')
    axes[1].axhline(z_idx, color='#f85149', alpha=0.5, lw=0.9, ls='--')
    axes[1].axvline(x_idx, color='#79c0ff', alpha=0.5, lw=0.9, ls='--')
    axes[2].axhline(z_idx, color='#f85149', alpha=0.5, lw=0.9, ls='--')
    axes[2].axvline(y_idx, color='#79c0ff', alpha=0.5, lw=0.9, ls='--')

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # HU Histogram
    st.markdown("---")
    with st.expander("📊 HU Histogram", expanded=False):
        fig_h, ax_h = plt.subplots(figsize=(10, 3), facecolor='#0d1117')
        ax_h.set_facecolor('#161b22')
        sample = vol.flatten()[::20]
        ax_h.hist(sample, bins=200, color='#1f6feb', alpha=0.85, edgecolor='none')
        ax_h.axvline(wc - ww/2, color='#f85149', lw=1.3, ls='--', label=f'lo={int(wc-ww/2)}')
        ax_h.axvline(wc + ww/2, color='#56d364', lw=1.3, ls='--', label=f'hi={int(wc+ww/2)}')
        ax_h.set_xlabel("HU", color='#c9d1d9')
        ax_h.set_ylabel("Count", color='#c9d1d9')
        ax_h.tick_params(colors='#8b949e')
        ax_h.legend(facecolor='#21262d', labelcolor='#c9d1d9', fontsize=9)
        for sp in ax_h.spines.values():
            sp.set_edgecolor('#30363d')
        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)

    # Slice strip
    st.markdown("---")
    st.markdown("#### 📽️ Axial Slice Strip")
    n_strip = st.slider("Bilangan slice", 4, 20, min(12, nz), key="strip")
    step_s  = max(1, nz // n_strip)
    idxs    = list(range(0, nz, step_s))[:n_strip]

    fig2, axes2 = plt.subplots(1, len(idxs), figsize=(2.2 * len(idxs), 2.8), facecolor='#0d1117')
    if len(idxs) == 1:
        axes2 = [axes2]
    for ax, idx in zip(axes2, idxs):
        ax.imshow(vol_w[idx, :, :], cmap=colormap, aspect='equal', origin='lower')
        if show_mask_2d and mask_vol is not None:
            m = mask_vol[idx].astype(float)
            ov = np.zeros((*m.shape, 4))
            ov[..., 0] = 1.0
            ov[..., 3] = np.where(m > 0, 0.40, 0.0)
            ax.imshow(ov, aspect='equal', origin='lower')
        border_col = '#f0883e' if idx == z_idx else 'none'
        ax.set_title(f"Z={idx}", color='#8b949e', fontsize=7, pad=2)
        ax.axis('off')
        ax.set_facecolor('#0d1117')
        for sp in ax.spines.values():
            sp.set_edgecolor(border_col)
            sp.set_linewidth(2)
            sp.set_visible(True)
    plt.tight_layout(pad=0.2)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


# ════════════════════════════════════════════════════════
# TAB 2 — 3D VOLUME RENDERING
# ════════════════════════════════════════════════════════
with tab_vol:
    st.markdown(f"#### 🌐 3D Volume Rendering — {selected_pid}")

    if not HAS_PLOTLY:
        st.error("❌ Plotly diperlukan:  `pip install plotly`")
    else:
        st.markdown(
            '<div class="info-box">⚡ Menggunakan <b>Isosurface rendering</b> — '
            'jauh lebih ringan daripada Volume rendering (elak Out of Memory).<br>'
            'Laraskan <b>ISO levels</b> untuk lihat lapisan yang berbeza.</div>',
            unsafe_allow_html=True,
        )

        # Hard cap: max 60 voxels per axis to avoid OOM
        max_dim = 60
        step = max(ds_step, nz // max_dim, ny // max_dim, nx // max_dim, 1)
        v_dn  = vol[::step, ::step, ::step].astype(np.float32)
        nzd, nyd, nxd = v_dn.shape

        # Build coordinate grid (in mm)
        zg = np.arange(nzd) * sz * step
        yg = np.arange(nyd) * sy * step
        xg = np.arange(nxd) * sx * step
        X, Y, Z = np.meshgrid(xg, yg, zg, indexing='ij')

        pcm = PLOTLY_CMAP.get(colormap, 'RdBu')

        # Isosurface levels: choose 3 meaningful HU values in the window
        lo_f = float(wc - ww / 2)
        hi_f = float(wc + ww / 2)
        iso_levels = [
            lo_f + (hi_f - lo_f) * 0.25,
            lo_f + (hi_f - lo_f) * 0.55,
            lo_f + (hi_f - lo_f) * 0.82,
        ]

        with st.spinner(f"Rendering isosurfaces… ({nxd}×{nyd}×{nzd} voxels, step={step})"):
            fig3 = go.Figure(data=go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=v_dn.flatten(),
                isomin=iso_levels[0],
                isomax=iso_levels[-1],
                surface_count=3,
                colorscale=pcm,
                opacity=0.25,
                showscale=True,
                colorbar=dict(
                    title=dict(text="HU", font=dict(color='#c9d1d9')),
                    tickfont=dict(color='#c9d1d9'),
                ),
                caps=dict(x_show=False, y_show=False, z_show=False),
            ))
            fig3.update_layout(
                scene=dict(
                    bgcolor='#0d1117',
                    xaxis=dict(backgroundcolor='#161b22', gridcolor='#30363d',
                               showbackground=True, title='X (mm)',
                               tickfont=dict(color='#8b949e')),
                    yaxis=dict(backgroundcolor='#161b22', gridcolor='#30363d',
                               showbackground=True, title='Y (mm)',
                               tickfont=dict(color='#8b949e')),
                    zaxis=dict(backgroundcolor='#161b22', gridcolor='#30363d',
                               showbackground=True, title='Z (mm)',
                               tickfont=dict(color='#8b949e')),
                    aspectmode='data',
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
                ),
                paper_bgcolor='#0d1117',
                margin=dict(l=0, r=0, b=0, t=40),
                title=dict(
                    text=(f"🌐 Isosurface — {selected_pid}  |  "
                          f"{nxd}×{nyd}×{nzd} voxels  |  step={step}×"),
                    font=dict(color='#c9d1d9', size=13),
                ),
                height=620,
            )

        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(
            '<div class="info-box">'
            '🖱️ <b>Drag</b> putar &nbsp;|&nbsp; 🔍 <b>Scroll</b> zoom &nbsp;|&nbsp;'
            ' 🖱️ <b>Right-click drag</b> pan &nbsp;|&nbsp; 🖱️ <b>Double-click</b> reset'
            '</div>', unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════
# TAB 3 — 3D SURFACE MESH
# ════════════════════════════════════════════════════════
with tab_msh:
    st.markdown(f"#### 🫀 3D Surface Mesh (Marching Cubes) — {selected_pid}")

    if not HAS_SKIMAGE:
        st.error("❌ scikit-image diperlukan:  `pip install scikit-image`")
    elif not HAS_PLOTLY:
        st.error("❌ Plotly diperlukan:  `pip install plotly`")
    else:
        # ── If masks available, show PROMINENT tip at top ──────────────────────
        if pdata['has_masks'] and mask_vol is not None:
            st.markdown("""
            <div style="background:#1a3a1a;border:1.5px solid #3fb950;border-radius:10px;
                        padding:12px 16px;margin-bottom:10px;font-size:13px;">
            ✅ <b>Mask segmentasi tersedia!</b> &nbsp;→&nbsp;
            Aktifkan <b>"Guna MASK sebagai permukaan"</b> di bawah untuk mendapatkan
            bentuk jantung yang tepat tanpa perlu tetapkan threshold.
            </div>""", unsafe_allow_html=True)

        # ── Basic controls ─────────────────────────────────────────────────────
        mc1, mc2, mc3 = st.columns(3)
        threshold  = mc1.number_input("Threshold HU", -500, 1500, iso_thresh, 10, key="mc_t")
        mc_step    = mc2.number_input("Downsample", 1, 8, ds_step, 1, key="mc_s")
        mesh_cmap  = mc3.selectbox("Colour", ['RdBu','Viridis','Plasma','Hot','Gray'], key="mcc")

        with st.expander("📖 Panduan HU (jantung dalam CT)"):
            st.markdown("""
| Struktur | HU biasa | Threshold cadangan |
|---|---|---|
| Udara / latar | -1000 – -500 | — |
| Lemak perikardium | -100 – -50 | — |
| Otot / myocardium | 50–80 | **60** |
| Darah (tanpa kontras) | 40–60 | **50** |
| **Darah (dengan kontras)** | 150–300 | **150 ← syor** |
| Kalsifikasi / tulang | 200–700 | 300 |

**Masalah biasa — bentuk nampak seperti tulang belakang/rusuk:**
- CT pesakit adalah CT dada penuh (bukan cardiac-only) — gunakan **ROI Crop** di bawah
  untuk hadkan ke kawasan tengah dada (jantung terletak di ~Z 30–70%, XY tengah)
- Atau aktifkan **Guna MASK** jika folder MASKS tersedia — ini cara terbaik
""")

        # ── Mask toggle ────────────────────────────────────────────────────────
        use_mask_for_mesh = False
        if pdata['has_masks'] and mask_vol is not None:
            use_mask_for_mesh = st.checkbox(
                "🩺 Guna MASK sebagai permukaan (lebih tepat — cadangan utama)",
                value=True, key="mc_mask"
            )

        # ── ROI Crop ───────────────────────────────────────────────────────────
        with st.expander("✂️ ROI Crop — hadkan kawasan jantung", expanded=not use_mask_for_mesh):
            st.markdown(
                "<small>Jantung biasanya berada di bahagian tengah dada. "
                "Crop boleh hapuskan tulang belakang & rusuk dari mesh.</small>",
                unsafe_allow_html=True,
            )
            cr1, cr2 = st.columns(2)
            z_lo_pct = cr1.slider("Z start %", 0, 90, 20, 5, key="z_lo",
                                   help="Buang slice bawah (kaki/perut)")
            z_hi_pct = cr1.slider("Z end %",   10, 100, 80, 5, key="z_hi",
                                   help="Buang slice atas (kepala/leher)")
            y_lo_pct = cr2.slider("Y start %", 0, 90, 15, 5, key="y_lo",
                                   help="Buang tepi anterior")
            y_hi_pct = cr2.slider("Y end %",   10, 100, 85, 5, key="y_hi",
                                   help="Buang tepi posterior")
            cx1, cx2 = st.columns(2)
            x_lo_pct = cx1.slider("X start %", 0, 90, 20, 5, key="x_lo",
                                   help="Buang tepi kiri")
            x_hi_pct = cx1.slider("X end %",   10, 100, 80, 5, key="x_hi",
                                   help="Buang tepi kanan")
            use_roi_crop = cx2.checkbox("Aktifkan ROI Crop", value=not use_mask_for_mesh,
                                         key="use_crop")

        # ── Connected-component filtering ──────────────────────────────────────
        with st.expander("🔗 Connected Component Filter — simpan hanya jantung"):
            st.markdown(
                "<small>Selepas Marching Cubes, mesh dibahagi kepada komponen terpisah. "
                "Simpan N komponen terbesar sahaja (jantung biasanya komponen paling besar "
                "di kawasan tengah dada).</small>",
                unsafe_allow_html=True,
            )
            if not HAS_SCIPY:
                st.warning("⚠️ scipy tidak dipasang — CC filter tidak tersedia.  "
                           "`pip install scipy`")
            cc1, cc2 = st.columns(2)
            use_cc_filter   = cc1.checkbox("Aktifkan CC filter",
                                            value=HAS_SCIPY, disabled=not HAS_SCIPY,
                                            key="use_cc")
            n_keep_cc       = cc2.number_input("Bilangan komponen disimpan", 1, 10, 1, 1,
                                                key="n_cc")

        if st.button("🔨 Generate 3D Mesh", use_container_width=True,
                     type="primary", key="gen_mesh"):
            with st.spinner("Menjalankan Marching Cubes…"):
                try:
                    from scipy import ndimage as _ndi

                    step_mc = int(mc_step)

                    if use_mask_for_mesh and mask_vol is not None:
                        # ── Use ground-truth mask directly ─────────────────────
                        vol_mc = mask_vol[::step_mc, ::step_mc, ::step_mc].astype(np.float32)
                        sp_mc  = (sz * step_mc, sy * step_mc, sx * step_mc)
                        thr_mc = 0.5
                        info_src = "Mask segmentasi (ground truth)"
                    else:
                        # ── CT HU with optional ROI crop ───────────────────────
                        raw = vol[::step_mc, ::step_mc, ::step_mc].astype(np.float32)

                        if use_roi_crop:
                            dnz, dny, dnx = raw.shape
                            z0 = int(dnz * z_lo_pct / 100); z1 = int(dnz * z_hi_pct / 100)
                            y0 = int(dny * y_lo_pct / 100); y1 = int(dny * y_hi_pct / 100)
                            x0 = int(dnx * x_lo_pct / 100); x1 = int(dnx * x_hi_pct / 100)
                            z1 = max(z1, z0 + 1); y1 = max(y1, y0 + 1); x1 = max(x1, x0 + 1)
                            raw = raw[z0:z1, y0:y1, x0:x1]
                            info_src = (f"CT HU >{float(threshold):.0f}  |  "
                                        f"Crop Z{z_lo_pct}-{z_hi_pct}% "
                                        f"Y{y_lo_pct}-{y_hi_pct}% "
                                        f"X{x_lo_pct}-{x_hi_pct}%")
                        else:
                            info_src = f"CT HU >{float(threshold):.0f}"

                        vol_mc = raw
                        sp_mc  = (sz * step_mc, sy * step_mc, sx * step_mc)
                        thr_mc = float(threshold)

                    # ── Marching Cubes ─────────────────────────────────────────
                    verts, faces, normals, vals = marching_cubes(
                        vol_mc, level=thr_mc, spacing=sp_mc, allow_degenerate=False
                    )

                    # ── Connected Component Filter ─────────────────────────────
                    n_removed = 0
                    if use_cc_filter and not use_mask_for_mesh:
                        try:
                            # Label voxels above threshold then filter faces
                            binary = (vol_mc >= thr_mc).astype(np.int8)
                            labeled, n_labels = _ndi.label(binary)
                            if n_labels > 1:
                                # Component sizes
                                sizes = np.bincount(labeled.ravel())
                                sizes[0] = 0  # background
                                # Keep top N largest
                                keep_labels = set(
                                    np.argsort(sizes)[::-1][:int(n_keep_cc)]
                                )
                                # Build mask of kept voxels
                                keep_mask = np.isin(labeled, list(keep_labels))
                                # Re-run marching cubes on filtered volume
                                vol_filt = np.where(keep_mask, vol_mc, thr_mc - 1.0)
                                verts2, faces2, _, _ = marching_cubes(
                                    vol_filt, level=thr_mc,
                                    spacing=sp_mc, allow_degenerate=False
                                )
                                n_removed = n_labels - int(n_keep_cc)
                                verts, faces = verts2, faces2
                                info_src += f"  |  CC filter: keep top {int(n_keep_cc)}/{n_labels}"
                        except Exception as _cc_err:
                            st.warning(f"CC filter dibuang (scipy error): {_cc_err}")

                    xv, yv, zv = verts.T
                    iv, jv, kv = faces.T
                    intensity   = (zv - zv.min()) / (zv.max() - zv.min() + 1e-9)

                    mesh_fig = go.Figure(data=[go.Mesh3d(
                        x=xv, y=yv, z=zv,
                        i=iv, j=jv, k=kv,
                        intensity=intensity,
                        colorscale=mesh_cmap,
                        opacity=float(mesh_opacity),
                        flatshading=False,
                        lighting=dict(
                            ambient=0.25, diffuse=0.85,
                            specular=0.65, roughness=0.35, fresnel=0.30,
                        ),
                        lightposition=dict(x=100, y=200, z=300),
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Z (mm)", font=dict(color='#c9d1d9')),
                            tickfont=dict(color='#c9d1d9'),
                        ),
                        name=f"Heart — {selected_pid}",
                    )])
                    mesh_fig.update_layout(
                        scene=dict(
                            bgcolor='#0d1117',
                            xaxis=dict(backgroundcolor='#161b22', gridcolor='#30363d',
                                       showbackground=True, title='X (mm)',
                                       tickfont=dict(color='#8b949e')),
                            yaxis=dict(backgroundcolor='#161b22', gridcolor='#30363d',
                                       showbackground=True, title='Y (mm)',
                                       tickfont=dict(color='#8b949e')),
                            zaxis=dict(backgroundcolor='#161b22', gridcolor='#30363d',
                                       showbackground=True, title='Z (mm)',
                                       tickfont=dict(color='#8b949e')),
                            aspectmode='data',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                        ),
                        paper_bgcolor='#0d1117',
                        margin=dict(l=0, r=0, b=0, t=55),
                        title=dict(
                            text=(
                                f"🫀 {selected_pid}  |  "
                                f"{len(verts):,} verts · {len(faces):,} faces  |  "
                                f"{info_src}"
                            ),
                            font=dict(color='#c9d1d9', size=12),
                        ),
                        height=660,
                    )

                    st.plotly_chart(mesh_fig, use_container_width=True)

                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Vertices", f"{len(verts):,}")
                    sc2.metric("Faces",    f"{len(faces):,}")
                    sc3.metric("Bounding box",
                               f"{xv.max()-xv.min():.0f}×{yv.max()-yv.min():.0f}×"
                               f"{zv.max()-zv.min():.0f} mm")
                    sc4.metric("Komponen dibuang", f"{n_removed}")

                    st.success(f"✅ Mesh berjaya! {len(verts):,} vertices · {len(faces):,} triangles")

                    # Download vertices CSV
                    with st.expander("💾 Export mesh vertices (CSV)"):
                        import io
                        buf = io.StringIO()
                        buf.write("x_mm,y_mm,z_mm\n")
                        for row in verts:
                            buf.write(f"{row[0]:.4f},{row[1]:.4f},{row[2]:.4f}\n")
                        st.download_button(
                            "⬇️ Download vertices.csv",
                            data=buf.getvalue(),
                            file_name=f"{selected_pid}_mesh_vertices.csv",
                            mime="text/csv",
                        )

                except Exception as exc:
                    st.error(f"❌ Marching Cubes gagal: {exc}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
                    st.info("Cuba: naikkan threshold, kurangkan downsample step, atau semak data.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:50px 20px;background:#161b22;
                        border-radius:12px;border:1px dashed #30363d;color:#8b949e">
                <div style="font-size:60px;margin-bottom:12px">🫀</div>
                <div style="font-size:18px;font-weight:600;color:#c9d1d9">
                    Klik <b>Generate 3D Mesh</b> untuk render</div>
                <div style="margin-top:10px;font-size:13px;line-height:1.8">
                    🩺 Cara terbaik: aktifkan <b>Guna MASK</b> (jika tersedia)<br>
                    ✂️ Alternatif: tetapkan <b>ROI Crop</b> + threshold <b>50–150 HU</b><br>
                    🔗 Aktifkan <b>CC Filter</b> untuk simpan jantung sahaja
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 4 — MASK OVERLAY  (only if masks available)
# ════════════════════════════════════════════════════════
if tab_msk is not None and mask_vol is not None:
    with tab_msk:
        st.markdown(f"#### 🩺 Segmentation Mask Overlay — {selected_pid}")

        st.markdown(
            '<div class="info-box">Mask (warna merah) menunjukkan kawasan jantung '
            'yang telah disegmentasi secara manual dalam dataset Rodriguez.</div>',
            unsafe_allow_html=True,
        )

        mz_idx = st.slider("Axial slice (Z)", 0, nz - 1, nz // 2, key="mz")
        mask_alpha_val = st.slider("Mask opacity", 0.1, 1.0, 0.5, 0.05, key="mkalpha")

        fig_m, axes_m = plt.subplots(1, 3, figsize=(16, 5.5), facecolor='#0d1117')
        fig_m.suptitle(
            f"Mask Overlay — {selected_pid}  |  Z={mz_idx}",
            color='#c9d1d9', fontsize=11, fontweight='bold',
        )

        mviews = [
            (vol_w[mz_idx, :, :], mask_vol[mz_idx, :, :], f"Axial Z={mz_idx}"),
            (vol_w[:, ny//2, :],  mask_vol[:, ny//2, :],  f"Coronal Y={ny//2}"),
            (vol_w[:, :, nx//2],  mask_vol[:, :, nx//2],  f"Sagittal X={nx//2}"),
        ]

        for ax, (img, msk, title) in zip(axes_m, mviews):
            ax.imshow(img, cmap='gray', aspect='equal', origin='lower')
            ov = np.zeros((*msk.shape, 4))
            ov[..., 0] = 1.0                              # red
            ov[..., 3] = np.where(msk > 0, mask_alpha_val, 0.0)
            ax.imshow(ov, aspect='equal', origin='lower')
            ax.set_title(title, color='#79c0ff', fontsize=11, fontweight='bold', pad=6)
            ax.tick_params(colors='#8b949e', labelsize=7)
            ax.set_facecolor('#0d1117')
            for sp in ax.spines.values():
                sp.set_edgecolor('#30363d')

        plt.tight_layout(pad=0.5)
        st.pyplot(fig_m, use_container_width=True)
        plt.close(fig_m)

        # Mask stats
        st.markdown("---")
        st.markdown("#### 📊 Mask Statistics")
        total_vox  = mask_vol.size
        heart_vox  = int(mask_vol.sum())
        heart_pct  = 100.0 * heart_vox / total_vox
        heart_vol_ml = heart_vox * sx * sy * sz / 1000.0  # mm³ → mL

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Heart voxels",  f"{heart_vox:,}")
        sc2.metric("Heart volume",  f"{heart_vol_ml:.1f} mL")
        sc3.metric("% of total",    f"{heart_pct:.2f}%")
        sc4.metric("Total voxels",  f"{total_vox:,}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🫀 3D Heart Viewer V2 &nbsp;|&nbsp; Rodriguez Benchmark Dataset &nbsp;|&nbsp; "
    "UTM Research — rohayanti@utm.my &nbsp;|&nbsp; "
    "DICOM parsing: {PatientID}_{Slice}_input.dcm"
)
