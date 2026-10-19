"""
Tests for merging the TotalSegmentator vessels into the model labels (vessels.py, CardioTwin step 6).

Synthetic and fast (no TotalSegmentator needed): the measured chambers keep
all their voxels, each vessel only takes the rule labels it may replace, the
rule SVC goes to the RA inside the heart, the aorta splits into the ascending
part (joins the aorta) and the descending aorta, and the new structures get
motion that fits them.
"""
import numpy as np
import pytest
import SimpleITK as sitk

import build_heart_model as bhm
import vessels as vs

SP = (1.0, 1.0, 1.0)
V = {name: v for v, name in vs.VESSEL_CLASSES.items()}


def _case():
    """Rule labels and a vessel map on a 30x40x40 grid, with deliberate overlaps."""
    rules = np.zeros((30, 40, 40), np.uint8)
    rules[5:15, 5:15, 5:15] = 1        # LV
    rules[5:15, 20:30, 5:15] = 2       # LA body
    rules[5:15, 5:15, 20:30] = 6       # RV
    rules[16:20, 5:15, 20:30] = 5      # RA
    rules[16:20, 20:30, 20:30] = 7     # PA
    rules[22:26, 30:34, 30:34] = 8     # rule 'SVC'
    rules[:, 34:38, :] = 4             # catch-all myocardium sheet
    ves = np.zeros_like(rules)
    ves[22:26, 30:34, 30:34] = V["heart"]                  # the rule SVC lies inside the heart
    ves[16:20, 26:30, 26:30] = V["superior_vena_cava"]     # true SVC: over PA and RA
    ves[5:15, 20:30, 5:9] = V["pulmonary_vein"]            # over the LA body: must not take it
    ves[:, 35:37, 16:20] = V["pulmonary_vein"]             # over the myocardium sheet: takes it
    ves[0:4, 0:4, 0:4] = V["inferior_vena_cava"]           # background
    ves[10:14, 5:9, 15:19] = V["aorta"]                    # ascending: touches the LV
    ves[0:30, 0:3, 36:39] = V["aorta"]                     # descending: away from the LV
    ves[5:15, 31:34, 5:15] = V["atrial_appendage_left"]    # background next to the LA
    return rules, ves


def test_measured_chambers_keep_all_their_voxels():
    rules, ves = _case()
    out = vs.merge(rules, ves, SP)
    for k in (1, 2, 6):
        assert np.array_equal(out[rules == k] == k, np.ones((rules == k).sum(), bool)), k


def test_each_vessel_only_takes_what_it_may_replace():
    rules, ves = _case()
    out = vs.merge(rules, ves, SP)
    svc = ves == V["superior_vena_cava"]
    assert (out[svc] == 8).all()                                       # SVC takes RA and PA voxels
    pv = ves == V["pulmonary_vein"]
    assert (out[pv & (rules == 2)] == 2).all()                         # ... the PV does not take the LA body
    assert (out[pv & (rules == 4)] == 10).all()                        # ... but takes the catch-all sheet
    assert (out[ves == V["inferior_vena_cava"]] == 9).all()
    assert (out[ves == V["atrial_appendage_left"]] == 2).all()         # appendage joins the LA


def test_rule_svc_goes_to_the_ra_and_the_aorta_splits():
    rules, ves = _case()
    out = vs.merge(rules, ves, SP)
    assert (out[rules == 8] == 5).all()                                # rule 'SVC' inside the heart -> RA
    assert (out[10:14, 5:9, 15:19] == 3).all()                         # ascending part joins the aorta
    assert (out[0:30, 0:3, 36:39] == 11).all()                         # descending aorta: its own structure


def test_with_deep_learning_chambers_vessels_only_fill_background():
    rules, ves = _case()
    out = vs.merge(rules, ves, SP, deep=True)
    for k in (1, 2, 4, 6, 7):                                          # every chamber / wall label is kept
        assert (out[rules == k] == k).all(), k
    assert (out[rules == 8] == 8).all()                                # label 8 is already TotalSegmentator's SVC
    assert (out[ves == V["inferior_vena_cava"]] == 9).all()            # background -> IVC
    pv = ves == V["pulmonary_vein"]
    assert (out[pv & (rules == 4)] == 4).all() and (out[pv & (rules == 0)] == 10).all()
    svc = ves == V["superior_vena_cava"]
    assert (out[svc & (rules == 5)] == 8).all() and (out[svc & (rules == 7)] == 7).all()   # SVC: RA junction only


def test_load_refuses_another_grid(tmp_path, monkeypatch):
    lab = sitk.GetImageFromArray(np.zeros((10, 10, 10), np.uint8))
    other = sitk.GetImageFromArray(np.zeros((10, 10, 12), np.uint8))
    import heart_io as hio
    monkeypatch.setattr(hio, "OUT_MODELS", str(tmp_path))
    assert vs.load(lab) is None                                        # not run yet
    sitk.WriteImage(other, vs.vessels_file())
    assert vs.load(lab, log=lambda *a: None) is None                    # made on another grid
    sitk.WriteImage(lab, vs.vessels_file())
    assert vs.load(lab).shape == (10, 10, 10)


class _Motion:
    """Stand-in CardiacMotion: each target returns a constant so the branch choice is visible."""
    def ventricular_systole(self, p, s_anchor=None):
        return np.full_like(p, 1.0)
    def atrial_systole(self, p, center=None):
        return np.full_like(p, 2.0)
    def aortic_distension(self, p, centre, dist=None):
        return np.full_like(p, 3.0)


def test_new_structures_move_sensibly():
    p = np.zeros((4, 3))
    T = lambda k, dao=None: bhm.motion_targets(_Motion(), k, p, None, None, None, None, None, dao)
    desc = T(11, dao=(None, None))
    assert not desc[0].any() and not desc[1].any() and (desc[2] == 3).all()     # only distends
    pv = T(10)
    assert (pv[0] == 1).all() and np.allclose(pv[1], 0.6) and not pv[2].any()   # rides with the LA
    assert not any(t.any() for t in T(9))                                      # IVC: held by the diaphragm
    assert np.allclose(T(8)[1], 0.6)                                            # SVC rides with the RA
    assert {9, 10, 11} <= set(bhm.MODEL_LABELS) and bhm.MODEL_SLUGS[11] == "desc_aorta"
