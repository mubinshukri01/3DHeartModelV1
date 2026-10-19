"""Shared DICOM loading helpers for the 3D heart pipeline."""
import os
import SimpleITK as sitk

# project root = folder above scripts/ (C:\3DHeartModelV1 locally, /var/www/3dheart on the VPS)
PROJECT = os.environ.get("HEART3D_ROOT",
                         os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(PROJECT, "dataset")
OUT_MODELS = os.path.join(PROJECT, "output", "3d_models")
OUT_VIS = os.path.join(PROJECT, "output", "visualizations")
DEFAULT_SERIES = "CT CCTA DL"


def load_series(folder):
    """Read a DICOM series as a SimpleITK image (HU), sorted by slice position."""
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(folder)
    if not ids:
        raise FileNotFoundError(f"No DICOM series in {folder}")
    # pick the series with most files
    files = max((reader.GetGDCMSeriesFileNames(folder, i) for i in ids), key=len)
    reader.SetFileNames(files)
    img = reader.Execute()
    return sitk.Cast(img, sitk.sitkFloat32)


def resample_iso(img, spacing=1.0, interp=sitk.sitkLinear, default=-1024):
    old_sp, old_sz = img.GetSpacing(), img.GetSize()
    new_sz = [int(round(sz * sp / spacing)) for sz, sp in zip(old_sz, old_sp)]
    return sitk.Resample(img, new_sz, sitk.Transform(), interp, img.GetOrigin(),
                         (spacing,) * 3, img.GetDirection(), default, img.GetPixelID())
