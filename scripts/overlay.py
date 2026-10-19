"""
==============================================================
  Segmentation overlays for the DICOM viewer (step 13)
  UTM-Claude Research Project | rohayanti@utm.my

  A label volume from the pipeline (NIfTI) is resampled in patient
  coordinates onto the pixel grid of the DICOM slice on screen. So it fits
  every reconstruction of the scan it was made on: CCTA, Lungs, Mediastinum
  and SSFreeze have other pixel sizes and fields of view.

  All CT series of TC_PTID share one frame of reference, but that does not
  make them the same scan. The calcium-score scan is a separate breath-hold
  (~4.6 mm off, 6 min earlier). A layer is therefore offered only on
  slices whose acquisition matches the series it was made on: same frame
  of reference and AcquisitionTime within SAME_SCAN_S. The epicardial fat
  has a version on each scan: measured on the calcium score (eat_cac) and
  mapped onto the CCTA (eat_labels).
==============================================================
"""
import functools, json, os

import numpy as np
import pydicom
import SimpleITK as sitk

import heart_io as hio
from segment_heart import LABELS

SAME_SCAN_S = 2.0            # reconstructions of one acquisition: AcquisitionTime within this many seconds
EAT_RGB = (1.0, 0.8, 0.1)    # yellow, as in the EAT QC figures
HEART = {k: (name, rgba[:3]) for k, (name, rgba) in LABELS.items()}
FAT = {1: ("Epicardial fat", EAT_RGB)}


def _stats(name):
    try:
        with open(os.path.join(hio.OUT_MODELS, name)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def layers():
    """Overlay layers on disk: {layer name: [(NIfTI path, source series, {value: (label name, rgb)}), ...]}.

    A layer can have one entry per scan; layer_for() picks the one made on the slice's scan.
    """
    seg, dl, eat = _stats("segmentation_stats.json"), _stats("dl_segmentation.json"), _stats("eat_stats.json")
    m = hio.OUT_MODELS
    out = {"Heart (rules)": [(os.path.join(m, "heart_labels.nii.gz"), seg and seg["series"], HEART)],
           "Heart (deep learning)": [(os.path.join(m, "heart_labels_dl.nii.gz"), dl and dl["series"], HEART)],
           "Epicardial fat": [(os.path.join(m, "eat_cac.nii.gz"), eat and eat["series"], FAT),        # measured here
                              (os.path.join(m, "eat_labels.nii.gz"), seg and seg["series"], FAT)]}  # mapped onto the CCTA
    out = {k: [e for e in v if e[1] and os.path.exists(e[0])] for k, v in out.items()}
    return {k: v for k, v in out.items() if v}


def _seconds(t):
    """DICOM time 'HHMMSS.ffffff' -> seconds since midnight."""
    t = str(t)
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:] or 0)


def same_scan(ds, ref):
    """True when two DICOM headers come from the same acquisition (not just the same patient position)."""
    try:
        return (ds.FrameOfReferenceUID == ref.FrameOfReferenceUID
                and abs(_seconds(ds.AcquisitionTime) - _seconds(ref.AcquisitionTime)) <= SAME_SCAN_S)
    except (AttributeError, ValueError):
        return False


@functools.lru_cache(maxsize=16)
def series_header(series, dataset=None):
    """Header of the first slice of a dataset series (the scan a layer was made on)."""
    folder = os.path.join(dataset or hio.DATASET, series)
    first = sorted(f for f in os.listdir(folder) if f.lower().endswith(".dcm"))[0]
    return pydicom.dcmread(os.path.join(folder, first), stop_before_pixels=True)


def layer_for(ds, entries):
    """The entry of a layer made on the scan of slice header `ds`, or None."""
    for entry in entries:
        try:
            if same_scan(ds, series_header(entry[1])):
                return entry
        except (OSError, IndexError):
            continue
    return None


@functools.lru_cache(maxsize=8)
def _load(path, mtime):
    return sitk.ReadImage(path)


def load_labels(path):
    """Label volume, cached until the file changes."""
    return _load(path, os.path.getmtime(path))


def label_slice(labels, slice_img):
    """Labels on the pixel grid of one plane: a DICOM slice read from its file, or a one-voxel-thick
    coronal / sagittal cut of a volume. Returns the 2D array without the thin axis."""
    out = sitk.Resample(labels, slice_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, labels.GetPixelID())
    a = sitk.GetArrayFromImage(out)
    thin = [i for i, n in enumerate(a.shape) if n == 1]
    return a.squeeze(axis=thin[0]) if a.ndim == 3 and thin else a


def rgba(lab, palette, alpha):
    """RGBA image of a label slice: each value in its colour at `alpha`, everything else transparent."""
    out = np.zeros(lab.shape + (4,), np.float32)
    for value, (_, rgb) in palette.items():
        m = lab == value
        out[m] = (*rgb, alpha)
    return out


def present(lab, palette, px_mm2):
    """[(label name, rgb, area in mm2)] of the labels visible in a slice, largest first."""
    rows = [(name, rgb, float((lab == value).sum() * px_mm2)) for value, (name, rgb) in palette.items()]
    return sorted((r for r in rows if r[2] > 0), key=lambda r: -r[2])
