"""
==============================================================
  Slice display for the DICOM viewer (step 14)
  UTM-Claude Research Project | rohayanti@utm.my

  * MPR: coronal and sagittal planes cut from a series volume, shown the
    radiological way (superior up, patient right on the viewer's left,
    sagittal with anterior on the left) and with the physical pixel
    aspect (e.g. 0.38 mm across, 0.625 mm between slices)
  * HU readout: each window is a small HTML canvas that carries the HU
    values. Hovering shows the HU, the structure under the cursor (with a
    segmentation overlay on) and the position in patient mm (LPS)
  * Position sync: a window can take the plane nearest to another
    window's plane in patient coordinates, whatever the slice spacing;
    planes in another direction get a dashed line where they cross

  Volumes are assumed axis-aligned (direction matrix diagonal, as for
  these axial CT series); signs are handled through SimpleITK's
  index <-> physical transforms.
==============================================================
"""
import base64, io, json
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from PIL import Image

VIEWS = {"Axial": 2, "Coronal": 1, "Sagittal": 0}      # view -> physical axis of its normal (x=0, y=1, z=2)
ORIENTATION = {"Axial": ("A", "P", "R", "L"), "Coronal": ("S", "I", "R", "L"),   # letters: top, bottom,
               "Sagittal": ("S", "I", "A", "P")}                                  # left, right of the image
MIN_MPR_SLICES = 10     # a folder needs this many slices for coronal / sagittal views


@dataclass
class Plane:
    """One displayed plane. Rows run top to bottom as shown; p0 + row*dr + col*dc = patient LPS (mm)."""
    hu: np.ndarray           # (rows, cols) HU as shown, or an RGB image (rows, cols, 3) without HU
    ref: object              # one-voxel-thick SimpleITK image of the plane (for resampling labels), or None
    flip: bool               # rows reversed relative to the reference image
    p0: np.ndarray
    dr: np.ndarray
    dc: np.ndarray
    axis: int                # physical axis of the plane's normal
    coord: float             # patient coordinate of the plane along that axis (mm)

    @property
    def spacing(self):
        return float(np.linalg.norm(self.dr)), float(np.linalg.norm(self.dc))


def _geometry(ref, index_of):
    """p0, dr, dc from a function (row, col) -> continuous index of `ref`."""
    pt = lambda r, c: np.array(ref.TransformContinuousIndexToPhysicalPoint(index_of(r, c)))
    p0 = pt(0, 0)
    return p0, pt(1, 0) - p0, pt(0, 1) - p0


def plane(vol, view, index):
    """Plane `index` (along the view's normal) of a volume, oriented for display."""
    a = sitk.GetArrayViewFromImage(vol)                   # (z, y, x)
    nz = vol.GetSize()[2]
    if view == "Axial":
        ref, hu, flip = vol[:, :, index:index + 1], a[index], False
        p0, dr, dc = _geometry(ref, lambda r, c: (c, r, 0))
    else:
        # superior at the top: reverse the rows when the slice index runs towards the head
        z0 = vol.TransformIndexToPhysicalPoint((0, 0, 0))[2]
        z1 = vol.TransformIndexToPhysicalPoint((0, 0, nz - 1))[2]
        flip = z1 > z0
        k = (lambda r: nz - 1 - r) if flip else (lambda r: r)
        if view == "Coronal":
            ref, hu = vol[:, index:index + 1, :], a[:, index, :]
            p0, dr, dc = _geometry(ref, lambda r, c: (c, 0, k(r)))
        else:
            ref, hu = vol[index:index + 1, :, :], a[:, :, index]
            p0, dr, dc = _geometry(ref, lambda r, c: (0, c, k(r)))
        hu = hu[::-1] if flip else hu
    axis = VIEWS[view]
    return Plane(np.ascontiguousarray(hu), ref, flip, p0, dr, dc, axis, float(p0[axis]))


def from_slice(arr, img):
    """Plane of one DICOM image read from its file (img: its SimpleITK image, or None without geometry).

    The normal comes from the geometry, so a localizer (scout) is not taken for an axial slice.
    """
    if img is None or img.GetDimension() != 3:
        z = np.zeros(3)
        return Plane(arr, None, False, z, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), 2, 0.0)
    p0, dr, dc = _geometry(img, lambda r, c: (c, r, 0))
    axis = int(np.argmax(np.abs(np.cross(dc, dr))))
    return Plane(arr, img, False, p0, dr, dc, axis, float(p0[axis]))


def labels_on(pl, labels_2d):
    """Labels resampled on pl.ref (overlay.label_slice), turned the way the plane is shown."""
    return labels_2d[::-1] if pl.flip else labels_2d


def nearest(vol, view, coord):
    """Index of the plane of `view` nearest to patient coordinate `coord` (mm), or None if outside the volume."""
    axis = VIEWS[view]
    pt = list(vol.TransformIndexToPhysicalPoint((0, 0, 0)))
    pt[axis] = coord
    i = vol.TransformPhysicalPointToContinuousIndex(pt)[axis]
    n = vol.GetSize()[axis]
    return int(round(i)) if -0.5 <= i <= n - 0.5 else None


def nearest_z(positions, coord, tolerance):
    """Index of the slice whose height is nearest to `coord`, or None if further than `tolerance` mm."""
    if not positions:
        return None
    i = int(np.argmin(np.abs(np.asarray(positions) - coord)))
    return i if abs(positions[i] - coord) <= tolerance else None


def crossing(pl, axis, coord):
    """Where a plane with normal `axis` at `coord` cuts this plane: ("row", r), ("col", c) or None."""
    if axis == pl.axis:
        return None
    for kind, step in (("row", pl.dr), ("col", pl.dc)):
        if abs(step[axis]) > 1e-6:
            i = (coord - pl.p0[axis]) / step[axis]
            n = pl.hu.shape[0 if kind == "row" else 1]
            return (kind, float(i)) if -0.5 <= i <= n - 0.5 else None
    return None


def compose(hu, ww, wl, cmap, labels=None, palette=None, alpha=0.45):
    """Windowed, colour-mapped image with the labels blended in: uint8 (rows, cols, 3)."""
    if hu.ndim == 3:                                        # RGB secondary capture: shown as is
        rgb = hu[..., :3].astype(np.float32) / (255.0 if hu.max() > 1 else 1.0)
    else:
        import matplotlib
        t = np.clip((hu.astype(np.float32) - (wl - ww / 2)) / max(ww, 1), 0, 1)
        rgb = matplotlib.colormaps[cmap](t)[..., :3].astype(np.float32)
    if labels is not None and palette:
        for value, (_, colour) in palette.items():
            m = labels == value
            rgb[m] = rgb[m] * (1 - alpha) + np.asarray(colour, np.float32) * alpha
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def html(rgb, pl, height, title="", footer="", orient=("", "", "", ""), labels=None, names=None, line=None):
    """Self-contained HTML canvas of one plane with the hover readout (for streamlit.components.v1.html)."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    rows, cols = rgb.shape[:2]
    sr, sc = pl.spacing
    meta = {"png": _b64(buf.getvalue()), "rows": rows, "cols": cols, "sr": sr or 1.0, "sc": sc or 1.0,
            "p0": pl.p0.tolist(), "dr": pl.dr.tolist(), "dc": pl.dc.tolist(), "geo": pl.ref is not None,
            "hu": _b64(np.clip(np.rint(pl.hu), -32768, 32767).astype("<i2").tobytes()) if pl.hu.ndim == 2 else None,
            "lab": _b64(labels.astype(np.uint8).tobytes()) if labels is not None else None,
            "names": {str(k): v for k, v in (names or {}).items()},
            "title": title, "footer": footer, "orient": list(orient), "line": list(line) if line else None}
    return _TEMPLATE.replace("__H__", str(int(height))).replace("__META__", json.dumps(meta))


_TEMPLATE = r"""<div id="w" style="position:relative;width:100%;height:__H__px;background:#111;border-radius:8px;
 overflow:hidden;font:11px system-ui,sans-serif;color:#eee;user-select:none">
<canvas id="c" style="position:absolute;cursor:crosshair"></canvas>
<div id="t" style="position:absolute;left:6px;top:6px;background:rgba(192,57,43,.9);padding:2px 6px;border-radius:4px"></div>
<div class="o" id="oT"></div><div class="o" id="oB"></div><div class="o" id="oL"></div><div class="o" id="oR"></div>
<div id="r" style="position:absolute;left:0;right:0;bottom:0;height:20px;line-height:20px;padding:0 8px;
 background:rgba(0,0,0,.75);white-space:nowrap;overflow:hidden;text-overflow:ellipsis"></div></div>
<style>.o{position:absolute;color:#f1c40f;font-weight:700;font-size:12px;text-shadow:0 0 3px #000}</style>
<script>
const M = __META__;
const w = document.getElementById('w'), c = document.getElementById('c'), x = c.getContext('2d');
const rd = document.getElementById('r');
const bytes = s => Uint8Array.from(atob(s), ch => ch.charCodeAt(0));
const hu = M.hu ? new Int16Array(bytes(M.hu).buffer) : null;
const lab = M.lab ? bytes(M.lab) : null;
document.getElementById('t').textContent = M.title;
rd.textContent = M.footer;
const img = new Image();
function layout() {
  const W = w.clientWidth, H = w.clientHeight - 20, wmm = M.cols * M.sc, hmm = M.rows * M.sr;
  const s = Math.min(W / wmm, H / hmm), cw = wmm * s, ch = hmm * s, L = (W - cw) / 2, T = (H - ch) / 2;
  Object.assign(c.style, {width: cw + 'px', height: ch + 'px', left: L + 'px', top: T + 'px'});
  const put = (id, txt, css) => { const e = document.getElementById(id); e.textContent = txt; Object.assign(e.style, css); };
  put('oT', M.orient[0], {left: (L + cw / 2 - 4) + 'px', top: (T + 2) + 'px'});
  put('oB', M.orient[1], {left: (L + cw / 2 - 4) + 'px', top: (T + ch - 16) + 'px'});
  put('oL', M.orient[2], {left: (L + 4) + 'px', top: (T + ch / 2 - 8) + 'px'});
  put('oR', M.orient[3], {left: (L + cw - 12) + 'px', top: (T + ch / 2 - 8) + 'px'});
}
function draw() {
  c.width = M.cols; c.height = M.rows;
  x.drawImage(img, 0, 0);
  if (M.line) {
    x.save(); x.strokeStyle = 'rgba(255,215,0,.85)'; x.lineWidth = Math.max(1, M.cols / 350);
    x.setLineDash([6 * x.lineWidth, 4 * x.lineWidth]); x.beginPath();
    const v = M.line[1] + 0.5;
    if (M.line[0] === 'row') { x.moveTo(0, v); x.lineTo(M.cols, v); } else { x.moveTo(v, 0); x.lineTo(v, M.rows); }
    x.stroke(); x.restore();
  }
}
function readout(col, row) {
  const i = row * M.cols + col, parts = [];
  if (hu) parts.push('HU ' + hu[i]);
  if (lab && lab[i] && M.names[lab[i]]) parts.push(M.names[lab[i]]);
  if (M.geo) {
    const p = [0, 1, 2].map(k => M.p0[k] + row * M.dr[k] + col * M.dc[k]);
    parts.push('x ' + p[0].toFixed(1) + ', y ' + p[1].toFixed(1) + ', z ' + p[2].toFixed(1) + ' mm (LPS)');
  }
  return parts.join('  ·  ');
}
c.addEventListener('mousemove', e => {
  const b = c.getBoundingClientRect();
  const col = Math.floor((e.clientX - b.left) / b.width * M.cols), row = Math.floor((e.clientY - b.top) / b.height * M.rows);
  if (col >= 0 && row >= 0 && col < M.cols && row < M.rows) rd.textContent = readout(col, row);
});
c.addEventListener('mouseleave', () => { rd.textContent = M.footer; });
window.readout = readout;          // for tests
img.onload = () => { draw(); layout(); };
img.src = 'data:image/png;base64,' + M.png;
addEventListener('resize', layout);
</script>"""
