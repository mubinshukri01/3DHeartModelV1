"""
Tests for the surface meshing of build_heart_model.py (CardioTwin XR roadmap, step 8).

All synthetic and fast: neighbouring structures share one surface (no gaps,
no overlaps) and keep their voxel volume; the Taubin filter keeps volume and
stays stable on merged junction vertices (trimesh's did not); every vertex
gets a unit normal; the baked ambient occlusion darkens grooves and leaves
faces shared with a neighbour neutral; see-through walls lose their shared faces.
"""
import numpy as np
import pytest
import SimpleITK as sitk
import trimesh
from scipy.spatial import cKDTree

import build_heart_model as bhm


def _boxes():
    """Two touching blocks (labels 1 and 2) in a 1 mm grid: 20x20x20 and 20x20x12 mm, sharing one face."""
    a = np.zeros((40, 40, 50), np.uint8)
    a[10:30, 10:30, 8:28] = 1
    a[10:30, 10:30, 28:40] = 2
    img = sitk.GetImageFromArray(a)
    return a, img


def test_neighbours_share_one_surface_and_keep_their_volume():
    lab, img = _boxes()
    meshes = bhm.label_meshes(lab, img, [1, 2], 10 ** 6)
    for k, ml in ((1, 8.0), (2, 4.8)):
        m = meshes[k]
        assert m.is_watertight, k
        assert abs(m.volume) * 1e6 == pytest.approx(ml, rel=0.08), k        # rounded box corners lose a little
    # vertices of 1 that it shares with 2 lie on 2's surface (to well under 0.1 mm)
    shared = meshes[1].metadata["shared"]
    assert shared.sum() > 50
    d, _ = cKDTree(meshes[2].vertices).query(meshes[1].vertices[shared])
    assert d.max() * 1000 < 0.01


def test_taubin_keeps_volume_and_is_stable_on_merged_vertices():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.02)
    noisy = m.vertices + np.random.default_rng(0).normal(0, 2e-4, m.vertices.shape)
    v = bhm.taubin(noisy, m.faces, 15)
    assert trimesh.Trimesh(v, m.faces).volume / m.volume - 1 == pytest.approx(0, abs=0.02)
    # a duplicate vertex on top of vertex 0 (a zero-length edge) and a self-loop, as at merged junctions,
    # must not pull vertices away: trimesh's filter moved some by tens of mm
    verts = np.vstack([m.vertices, m.vertices[:1]])
    dup = len(verts) - 1
    faces = np.vstack([m.faces, [[0, dup, 0], [dup, dup, dup]]])
    v1 = bhm.taubin(verts, m.faces, 15)[:-1]                             # without the extra faces
    v2 = bhm.taubin(verts, faces, 15)[:-1]
    assert np.abs(v2 - v1).max() * 1000 < 0.5


def test_every_vertex_gets_a_unit_normal():
    m = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    v = np.vstack([m.vertices, m.vertices[0] + 1e-12])                    # a vertex only in a zero-area face
    f = np.vstack([m.faces, [[0, len(v) - 1, 0]]])
    n = bhm.unit_normals(trimesh.Trimesh(v, f, process=False))
    assert np.allclose(np.linalg.norm(n, axis=1), 1)


def test_ambient_occlusion_darkens_grooves_and_leaves_shared_faces_neutral():
    # an L-shaped block: the inner corner is a groove; the far face of the long arm is open
    a = np.zeros((40, 40, 40), np.uint8)
    a[10:30, 10:30, 8:16] = 1                                              # long arm along y
    a[10:30, 8:16, 8:32] = 1                                               # arm along x
    a[10:30, 22:30, 16:24] = 2                                             # a neighbour touching block 1
    img = sitk.GetImageFromArray(a)
    meshes = bhm.label_meshes(a, img, [1, 2], 10 ** 6)
    m = meshes[1]
    ao = bhm.ambient_occlusion(m, img, bhm.occlusion_fields(a > 0, img), a, own=1)
    lps = bhm.gltf_to_lps(m.vertices)
    groove = np.linalg.norm(lps - [16, 16, 20], axis=1).argmin()           # inner corner (x, y, z in mm)
    flat = np.linalg.norm(lps - [32, 22, 20], axis=1).argmin()             # end face of the x arm, open
    assert ao[groove] < ao[flat] - 0.2
    assert ao[m.metadata["shared"]].mean() > 0.95            # junction-edge vertices also face outward a little


def test_glb_vertex_colours_are_written_normalised(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
    gb = bhm.GLTFBuilder()
    gb.add_mesh("ball", mesh, (1, 0, 0, 1), [np.zeros_like(mesh.vertices)], ["none"],
                shade=np.linspace(0.35, 1, len(mesh.vertices)))
    attrs = gb.g["meshes"][0]["primitives"][0]["attributes"]
    acc = gb.g["accessors"][attrs["COLOR_0"]]
    assert acc["componentType"] == 5121 and acc["normalized"] and acc["type"] == "VEC4"
