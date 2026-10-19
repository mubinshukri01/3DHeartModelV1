"""
==============================================================
  Rodriguez cardiac-fat benchmark: loading, alignment, ground truth
  UTM-Claude Research Project | rohayanti@utm.my

  dataset/Rodriguez/ holds 20 non-contrast calcium-score CTs (Philips
  iCT 256 / Siemens Sensation 64, 3 mm slices), one flat folder each:
    DICOM/{ID}_{nnn}_input.dcm   original slices
    IMAGES/{ID}_{nnn}_input.bmp  fat-window image: HU -200..-30 -> 1.5*(HU+200)
    MASKS/{ID}_{nnn}_mask.bmp    the same image with each fat pixel coloured
                                 by class: red = epicardial fat (EAT),
                                 green = mediastinal fat, blue = pericardium
                                 line, grey = unlabelled fat

  The BMPs are NOT pixel-aligned with the DICOM. Each patient was shifted,
  and scans with another pixel size were resampled to 0.3515625 mm.
  register() recovers that scale + shift from the images themselves.
  load_patient() then returns the CT resampled onto the annotation grid,
  so a segmentation can be scored against the masks voxel by voxel.
==============================================================
"""
import glob, os
import numpy as np
import pydicom
from PIL import Image
from scipy import ndimage as ndi

import heart_io as hio

ROOT = os.path.join(hio.DATASET, "Rodriguez")
FAT_RANGE = (-200, -30)      # HU window the annotators used for the fat images
BMP_PX = 0.3515625           # mm, common pixel size of the resampled annotations
SLICE_MM = 3.0
MIN_SLICES = 10              # TJes has a single slice and is left out of the benchmark


def patient_ids(root=ROOT, min_slices=MIN_SLICES):
    counts = {}
    for f in glob.glob(os.path.join(root, "DICOM", "*_input.dcm")):
        pid = os.path.basename(f).split("_")[0]; counts[pid] = counts.get(pid, 0) + 1
    return sorted(p for p, n in counts.items() if n >= min_slices)


def to_grid(img, scale, ty, tx, order=1, cval=-1024.0):
    """Resample a DICOM-space slice onto the 512x512 annotation grid (x_bmp = scale * x_dcm + t)."""
    return ndi.affine_transform(img, np.diag([1 / scale, 1 / scale]), offset=[-ty / scale, -tx / scale],
                                output_shape=(512, 512), order=order, cval=cval)


def fat_image(hu):
    """Reproduce the dataset's fat-window BMP from HU."""
    lo, hi = FAT_RANGE
    return np.where((hu >= lo) & (hu <= hi), (hu - lo) * 1.5, 0.0)


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def register(hu, bmp, px):
    """Find scale + shift mapping a DICOM slice onto its BMP. Returns (scale, ty, tx, ncc).

    The scale is 1 or px / BMP_PX (the only two the dataset uses). The shift comes from
    FFT cross-correlation, refined to 0.25 px. NCC is 1.0 for unresampled patients and
    ~0.9 for resampled ones (the authors used a different interpolation kernel).
    """
    best = None
    for s in sorted({1.0, round(px / BMP_PX, 4)}):
        t0 = (1 - s) * 255.5                                    # keeps the image centre fixed
        a = fat_image(to_grid(hu, s, t0, t0))
        fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(bmp - bmp.mean())
        cc = np.fft.fftshift(np.real(np.fft.ifft2(fb * np.conj(fa))))
        r = 150; win = cc[256 - r:256 + r + 1, 256 - r:256 + r + 1]
        iy, ix = np.unravel_index(win.argmax(), win.shape)
        cur = (_ncc(fat_image(to_grid(hu, s, t0 + iy - r, t0 + ix - r)), bmp), t0 + iy - r, t0 + ix - r)
        for step in (0.5, 0.25):
            for _ in range(3):
                cand = [(_ncc(fat_image(to_grid(hu, s, cur[1] + dy, cur[2] + dx)), bmp), cur[1] + dy, cur[2] + dx)
                        for dy in (-step, 0, step) for dx in (-step, 0, step)]
                cur = max(cand + [cur])
        if best is None or cur[0] > best[3]:
            best = (s, cur[1], cur[2], cur[0])
    return best


def decode_mask(rgb):
    """Split a colour mask into (EAT ground truth, don't-care) boolean images.

    Each labelled fat pixel stores its brightness in its class channel, and the
    other channels carry some blending, so the class is the dominant channel.
    Pixels on the drawn pericardium line (blue) are neither EAT nor background:
    they are excluded from scoring.
    """
    rgb = rgb.astype(int)
    grey = (rgb[..., 0] == rgb[..., 1]) & (rgb[..., 1] == rgb[..., 2])
    dom = rgb.argmax(-1)
    return ~grey & (dom == 0), ~grey & (dom == 2)


def load_patient(pid, root=ROOT):
    """CT + annotations of one patient, all on the annotation grid.

    Returns dict with hu (Z,512,512 float32, HU on the annotation grid), fat
    (the annotators' fat pixels), eat (ground truth), ignore (pericardium line),
    px (annotation pixel size, mm), z_mm, slices, reg (scale, ty, tx, ncc).
    """
    ns = sorted(int(os.path.basename(f).split("_")[1]) for f in glob.glob(os.path.join(root, "DICOM", f"{pid}_*_input.dcm")))
    hus, bmps, eats, igns = [], [], [], []
    for n in ns:
        ds = pydicom.dcmread(os.path.join(root, "DICOM", f"{pid}_{n:03d}_input.dcm"))
        hus.append(ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept))
        px = float(ds.PixelSpacing[0])
        bmps.append(np.asarray(Image.open(os.path.join(root, "IMAGES", f"{pid}_{n:03d}_input.bmp")), dtype=np.float32))
        eat, ign = decode_mask(np.asarray(Image.open(os.path.join(root, "MASKS", f"{pid}_{n:03d}_mask.bmp")).convert("RGB")))
        eats.append(eat); igns.append(ign)
    mid = len(ns) // 2
    reg = register(hus[mid], bmps[mid], px)
    s, ty, tx, _ = reg
    return dict(hu=np.stack([to_grid(h, s, ty, tx) for h in hus]).astype(np.float32),
                fat=np.stack(bmps) > 0, eat=np.stack(eats), ignore=np.stack(igns),
                px=px / s, z_mm=SLICE_MM, slices=np.array(ns), reg=np.array(reg))
