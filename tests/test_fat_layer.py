"""
Tests for the epicardial-fat display layer of build_heart_model.py (CardioTwin XR roadmap, step 9).

Synthetic and fast: the layer is CCTA fat that touches the heart near the
measured EAT (not fat beyond the pericardium, far from the EAT, or inside
the heart); the fat's inner faces towards the heart are found; and in the
layered see-through rule the wall loses only the faces it shares with opaque
structures, while the fat (outer layer) also loses the faces it shares with the wall.
"""
import numpy as np
import SimpleITK as sitk
import trimesh

import build_heart_model as bhm

SP = (1.0, 1.0, 1.0)


def test_fat_layer_keeps_fat_on_the_heart_near_the_measured_eat():
    labels = np.zeros((30, 40, 50), np.uint8)
    labels[5:25, 10:30, 10:20] = 1                         # heart (x 10-19)
    ct = np.full(labels.shape, 40.0, np.float32)           # soft tissue
    ct[labels > 0] = 150.0
    ct[5:25, 10:30, 20:26] = -100                          # A: fat on the heart (x 20-25)
    ct[5:25, 10:30, 26:28] = 40                            # pericardium line
    ct[5:25, 10:30, 28:34] = -100                          # B: fat beyond it, still near the measured EAT
    ct[5:25, 10:30, 4:10] = -100                           # C: fat on the other side of the heart, far from the EAT
    ct[15, 20, 15] = -100                                  # a fat-like voxel inside the heart
    eat = np.zeros(labels.shape, bool)
    eat[5:25, 10:30, 20:34] = True                         # measured EAT (mapped) covers A and B
    fat = bhm.fat_layer(eat, ct, labels, SP)
    assert fat[8:22, 13:27, 21:25].all()                   # A kept
    assert not fat[:, :, 29:34].any()                      # B: not connected to the heart
    assert not fat[:, :, 4:10].any()                       # C: more than FAT_GUIDE_MM from the measured EAT
    assert not (fat & (labels > 0)).any()                  # never inside the heart


def test_faces_towards_the_heart_are_found():
    labels = np.zeros((30, 30, 30), np.uint8)
    labels[10:20, 10:20, 10:20] = 1                        # heart cube, x 10-19 (LPS mm = index)
    img = sitk.GetImageFromArray(labels)
    p = np.array([[22.0, 12, 12], [22, 16, 12], [22, 12, 16]])     # a triangle 2 mm beyond the cube, normal +x
    tri = bhm.lps_to_gltf(np.vstack([p, p]))
    m = trimesh.Trimesh(tri, [[0, 1, 2], [3, 5, 4]], process=False)   # second face: same place, normal -x
    hit = bhm.faces_towards(m, labels, img, {1}, 3.0)
    assert hit.tolist() == [False, True]                    # only the face looking back at the heart


def test_layered_see_through_rule():
    labels = np.zeros((40, 40, 44), np.uint8)
    labels[10:30, 10:30, 10:30] = 4                        # wall (see-through)
    labels[14:26, 14:26, 14:26] = 1                        # LV inside it (opaque)
    labels[10:30, 10:30, 30:35] = bhm.EAT_LABEL            # fat on one side of the wall (outer see-through layer)
    img = sitk.GetImageFromArray(labels)
    order = [4, 1, bhm.EAT_LABEL]
    meshes = bhm.label_meshes(labels, img, order, 10 ** 6)
    bit = {k: 1 << i for i, k in enumerate(order)}

    def common(m):
        return np.bitwise_and.reduce(m.metadata["neighbours"][m.faces], axis=1)

    wall, fat = meshes[4], meshes[bhm.EAT_LABEL]
    hidden = bhm.hidden_faces(wall, 4, order)
    assert hidden[common(wall) == bit[1]].all()                       # wall on the LV: dropped
    assert (common(wall) == bit[bhm.EAT_LABEL]).any()
    assert not hidden[common(wall) == bit[bhm.EAT_LABEL]].any()       # wall under the fat: kept (no hole)
    assert bhm.hidden_faces(fat, bhm.EAT_LABEL, order)[common(fat) == bit[4]].all()   # fat on the wall: dropped
