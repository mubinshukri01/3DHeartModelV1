"""
==============================================================
  STEP 2 - Surface meshes + animated cardiac cycle -> glTF / GLB
  UTM-Claude Research Project | rohayanti@utm.my

  Reads heart_labels.nii.gz (from segment_heart.py) and builds:
    * one smoothed mesh per structure (marching cubes + Taubin smoothing);
      the coronary arteries (coronaries.py), when present, at the CT's
      native resolution
    * procedural cardiac motion as glTF MORPH TARGETS:
        T0 ventricular systole  - the cavity contracts; the wall keeps its
                                  volume (thickens as it shortens); the
                                  base descends toward the apex; LV twist.
                                  The atria and great arteries stretch:
                                  their floor follows the AV plane, their
                                  top is held by the veins and the arch
        T1 atrial systole       - atrial contraction + late LV filling
        T2 great-artery distension - aorta and pulmonary artery expand
                                  during ejection
    Each structure has its own motion (motion_targets: the RV free wall
    contracts towards the RV axis, the RA and SVC around the RA centre, the
    IVC and descending aorta are held). Neighbours' motions are blended
    (motion_blend), so shared surfaces move together; the fat and the
    coronary arteries move with what they lie on.
    * a looping animation of the morph weights (Wiggers-diagram timing)

  The CCTA was reconstructed at 75% R-R (mid-diastole / diastasis),
  so the segmented shape is used as the rest pose (all weights = 0).
  The dataset has a single cardiac phase, so motion is procedural.
  With a multi-phase (4D) CCTA, the same morph-target slots can be filled
  with real per-phase displacements instead (see README_3D.md).

  Usage:
    python build_heart_model.py [--bpm 72] [--faces 450000]

  Output (output/3d_models):
    heart_animated.glb           single-file, for web / Unity (glTFast, UniGLTF)
    heart_animated.gltf + .bin   same model, JSON + binary
    heart_<structure>.stl/.obj   static meshes for printing / CAD
    heart_viewer.html            standalone three.js viewer (open in browser
                                 via  python -m http.server)
==============================================================
"""
import argparse, base64, json, os, struct, time
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from skimage import measure

import coronaries
import heart_io as hio
import vessels as vs
from segment_heart import LABELS, SLUGS

# structures of the 3D model: the rule labels, plus the TotalSegmentator vessels when vessels.py has run
MODEL_LABELS = {**LABELS, **vs.VESSEL_LABELS}
MODEL_SLUGS = {**SLUGS, **vs.VESSEL_SLUGS}

TARGETS = ["ventricular_systole", "atrial_systole", "great_artery_distension"]
LABEL_SIGMA_MM = 1.0        # smoothing of each structure's indicator before the shared-surface meshing
MC_LEVEL = 1e-4             # just off 0: exact zeros on grid points make degenerate marching-cubes triangles
JOINT_SMOOTH_ITERATIONS = 40    # Taubin half-steps over all structures at once (removes the voxel staircase);
                                # 80 looked the same; more z smoothing broke thin sheets up (step 8 review)
TAUBIN_NU = 0.53            # trimesh's inflate step (positive: trimesh subtracts it); with lambda 0.5 ~volume-neutral
AO_RADII_MM = (2.0, 4.0, 8.0)   # ambient occlusion: blur radii of the heart occupancy, sampled 1.5 radii out
AO_FLAT = 0.067             # occupancy sampled 1.5 sigma outside a flat surface (normal CDF at -1.5)
AO_STRENGTH = 1.6           # how much darker a fully enclosed groove gets
AO_MIN = 0.35               # darkest ambient-occlusion factor
INNER_SHARE = 0.05          # structures with < 5 % of their boundary on background (LV) only shade themselves
MOTION_SIGMA_MM = 3.0       # neighbouring structures' motion blends over ~2 sigma, so shared surfaces move together
ANCHOR_MM = 40.0            # the AV-plane descent fades out this far above the mitral plane: the veins hold the
                            # atrial roof (step 11: the model dragged the atria down), so the atria stretch and fill
ARTERY_ANCHOR_MM = 70.0     # ... and this far for the great arteries, held at the arch and the lung hila
WALL_MM = 10.0              # LV wall thickness that conserves its volume in systole (thickens as it shortens)
CAVITY_BINS = (24, 16)      # LV cavity radius fitted per (height, angle) bin around the long axis

# optional structure: epicardial fat (eat_patient.py). The measured EAT volume is in eat_stats.json; the model
# shows a display layer (fat_layer) that fits the 75 % CCTA heart
EAT_NAME, EAT_SLUG = "Epicardial fat (EAT)", "eat"
EAT_RGBA = (1.0, 0.95, 0.66, 0.6)    # pale cream: distinct from the golden aorta
EAT_LABEL = 12             # its label in the model labels (after the vessels, 9-11)
EAT_MIN_BLOB_VOX = 20       # drop specks smaller than this
FAT_HU = (-190.0, -30.0)    # fat window, as for the EAT measurement
FAT_CT_SIGMA_MM = 0.5       # light smoothing of the CCTA before the fat window (fewer specks)
FAT_GUIDE_MM = 4.0          # display fat lies within this distance of the measured EAT mapped from the 45 % scan
                            # (~1 mm residual registration + ~3 mm 45 %-vs-75 % surface motion, step 11)
FAT_TOUCH_MM = 3.0          # 'in contact with the heart': the 110 HU heart outline stops 1-2 mm short of the fat
MODEL_LABELS[EAT_LABEL] = (EAT_NAME, EAT_RGBA)
MODEL_SLUGS[EAT_LABEL] = EAT_SLUG
LAYER = {EAT_LABEL: 2}      # outer see-through layers (others 1): a layer drops faces shared with lower ones
EAT_AO_SCALE = 0.5         # half the ambient occlusion on the translucent fat: dark crevices read as dirt there

# optional structure: the coronary arteries (coronaries.py, TotalSegmentator's licensed coronary_arteries task).
# They are 2-4 mm wide, so they are meshed on their own from the native-resolution mask (coronary_mesh); on the
# 1 mm grid they are a label (slice tool, pointer, case package) that moves with what it lies on
COR_NAME, COR_SLUG = "Coronary arteries", "coronaries"
COR_RGBA = (0.95, 0.30, 0.15, 1.0)     # orange-red: distinct from the LV and the pulmonary veins
COR_LABEL = 13
COR_SIGMA_MM = 0.35         # smoothing of the mask before meshing: TotalSegmentator predicts on a 0.7 mm grid and
                            # copies that onto the CT's (0.38 x 0.38 x 0.63 mm), which left 0.7 mm beads at 0.25 mm
COR_LEVEL = 0.4             # surface of the smoothed mask: keeps the thin branches whole (12 pieces, 2.3 mL of 2.2);
                            # the same smoothing and level put them on the 1 mm grid
COR_SMOOTH_ITERATIONS = 30  # Taubin half-steps on their mesh
COR_FACE_SHARE = 0.1        # their face budget, as a share of total_faces (above it: marching cubes at 2 voxels)
MODEL_LABELS[COR_LABEL] = (COR_NAME, COR_RGBA)
MODEL_SLUGS[COR_LABEL] = COR_SLUG
CARRIED = (EAT_LABEL, COR_LABEL)    # no motion of their own: they move with the structure they lie on


def load_eat(lab_img, log=print):
    """EAT mask on the heart_labels grid, or None if eat_patient.py has not been run for this grid."""
    path = os.path.join(hio.OUT_MODELS, "eat_labels.nii.gz")
    if not os.path.exists(path):
        return None
    img = sitk.ReadImage(path)
    if img.GetSize() != lab_img.GetSize() or not (np.allclose(img.GetSpacing(), lab_img.GetSpacing())
                                                  and np.allclose(img.GetOrigin(), lab_img.GetOrigin())):
        log("  eat_labels.nii.gz is on another grid (re-run eat_patient.py) - skipping EAT")
        return None
    eat = sitk.GetArrayFromImage(img) > 0
    lab, n = ndi.label(eat)
    keep = np.flatnonzero(ndi.sum(eat, lab, range(1, n + 1)) >= EAT_MIN_BLOB_VOX) + 1
    return np.isin(lab, keep)

def fat_layer(eat, ct, labels, spacing_zyx):
    """Display layer of the epicardial fat on the 75 % CCTA grid (bool volume).

    The EAT was measured on the 45 % calcium-score scan; mapped onto the CCTA it overlaps the heart (13 of
    70 mL) and lies mostly on non-fat tissue (only 26 mL is fat there), so it looked patchy. Shown instead:
    the CCTA's own fat (FAT_HU) outside the heart, within FAT_GUIDE_MM of the mapped EAT, in pieces that
    touch the heart (epicardial fat is in contact with it; fat beyond the pericardium line is not connected).
    The measured volume stays the one in eat_stats.json.
    """
    sp = np.asarray(spacing_zyx, float)
    hu = ndi.gaussian_filter(ct.astype(np.float32), FAT_CT_SIGMA_MM / sp)
    cand = (hu >= FAT_HU[0]) & (hu <= FAT_HU[1]) & (labels == 0)
    cand &= ndi.distance_transform_edt(~eat, sampling=sp) <= FAT_GUIDE_MM
    lab, n = ndi.label(cand)
    touch = cand & (ndi.distance_transform_edt(labels == 0, sampling=sp) <= FAT_TOUCH_MM)
    keep = np.unique(lab[touch])
    keep = keep[keep > 0]
    keep = keep[ndi.sum(cand, lab, keep) >= EAT_MIN_BLOB_VOX] if len(keep) else keep
    return np.isin(lab, keep)


def coronary_grid(cor, lab_img):
    """The coronary mask (native grid, coronaries.py) on the heart_labels grid (bool)."""
    f = sitk.SmoothingRecursiveGaussian(sitk.Cast(cor, sitk.sitkFloat32), COR_SIGMA_MM)
    return sitk.GetArrayFromImage(sitk.Resample(f, lab_img, sitk.Transform(), sitk.sitkLinear, 0.0)) >= COR_LEVEL


def coronary_mesh(cor, target_faces):
    """Surface of the coronary mask on its native grid (glTF space)."""
    mask = sitk.GetArrayFromImage(cor) > 0
    return mask_to_mesh(mask, cor, target_faces, sigma=COR_SIGMA_MM / np.array(cor.GetSpacing())[::-1],
                        level=COR_LEVEL, iterations=COR_SMOOTH_ITERATIONS)

# ── mesh ──────────────────────────────────────────────────────────────────
def lps_to_gltf(p):
    """Patient LPS (mm) -> glTF (m): +X = patient left, +Y = superior, +Z = anterior."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1) / 1000.0


def mask_to_mesh(mask, img, target_faces, sigma=1.0, level=0.5, iterations=15):
    """One structure's closed surface (glTF space): the mask smoothed by sigma (voxels; one value or (z, y, x)),
    cut at level, then `iterations` Taubin half-steps."""
    pad = np.pad(mask.astype(np.float32), 2)
    sm = ndi.gaussian_filter(pad, sigma)
    # coarser marching-cubes step until the face budget is met (no extra deps)
    for step in (1, 2, 3, 4):
        verts, faces, _, _ = measure.marching_cubes(sm, level, step_size=step)
        if len(faces) <= 1.3 * target_faces:
            break
    verts -= 2  # remove pad (index space, z,y,x)
    # index (z,y,x) -> physical LPS (direction assumed axis-aligned)
    sp, org = np.array(img.GetSpacing()), np.array(img.GetOrigin())
    D = np.array(img.GetDirection()).reshape(3, 3)
    lps = (verts[:, ::-1] * sp) @ D.T + org
    m = trimesh.Trimesh(lps_to_gltf(lps), faces[:, ::-1], process=True)
    # trimesh's nu is positive (it is subtracted); nu=-0.53 made both half-steps shrink (plain Laplacian):
    # meshes came out 4-22 % too small until 9 Oct 2026
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=TAUBIN_NU, iterations=iterations)
    # (near-)zero-area faces, which marching cubes at step > 1 leaves, have no face normal; a vertex surrounded
    # by them gets a zero-length normal, which glTF forbids (Khronos validator: ACCESSOR_VECTOR3_NON_UNIT).
    # They add no area or volume, so drop them (nondegenerate_faces() uses another threshold and keeps some)
    m.update_faces(np.linalg.norm(m.face_normals, axis=1) > 0.5)
    m.remove_unreferenced_vertices()
    m.fix_normals()
    return m


def gltf_to_lps(p):
    """glTF (m) -> patient LPS (mm); inverse of lps_to_gltf (also for displacements and normals)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1) * 1000.0


def label_meshes(labels, img, ids, total_faces, sigma_mm=None):
    """Surfaces of several structures that fit together: {label: mesh in glTF space}.

    Each structure's indicator is smoothed, and its surface is where that smoothed indicator equals the
    largest one of the others (background included). Two neighbours therefore share one surface, with no
    gaps or overlaps between them. Smoothing each mask on its own (mask_to_mesh) left up to 5 mm gaps and
    1-2 mm overlaps at the joins and shrank thin structures by up to 22 %. All structures use one
    marching-cubes step (1, or 2 above the face budget) so the shared surfaces stay matched.
    """
    sigma_mm = LABEL_SIGMA_MM if sigma_mm is None else sigma_mm
    pad = 2
    p = np.pad(labels, pad)
    # one value, or (z, y, x) in mm: the staircase runs along the CT slices, so z can be smoothed more
    sig = np.broadcast_to(np.asarray(sigma_mm, float), (3,)) / np.array(img.GetSpacing()[::-1])
    S = np.stack([ndi.gaussian_filter((p == k).astype(np.float32), sig) for k in [0] + list(ids)])
    best, top1 = S.argmax(0), S.max(0)
    rest = S.copy()
    np.put_along_axis(rest, best[None], -1.0, 0)
    top2 = rest.max(0)
    del rest
    sp, org = np.array(img.GetSpacing()), np.array(img.GetOrigin())
    D = np.array(img.GetDirection()).reshape(3, 3)
    for step in (1, 2):
        out = {}
        for i, k in enumerate(ids, start=1):
            field = S[i] - np.where(best == i, top2, top1)      # > 0 where k beats every other structure
            verts, faces, _, _ = measure.marching_cubes(field, MC_LEVEL, step_size=step)
            lps = ((verts - pad)[:, ::-1] * sp) @ D.T + org
            m = trimesh.Trimesh(lps_to_gltf(lps), faces[:, ::-1], process=True)
            m.fix_normals()
            out[k] = m
        if sum(len(m.faces) for m in out.values()) <= 1.3 * total_faces:
            break
    return smooth_jointly(out)


def smooth_jointly(meshes, iterations=None, tolerance_m=2e-6):
    """Taubin smoothing of all structures as one connected mesh (removes the 1 mm voxel staircase).

    Vertices that two structures share (their common surface, equal up to MC_LEVEL) are merged first, so
    a shared surface moves as one and the structures stay matched; smoothing each mesh on its own would
    pull them apart again. Taubin (lambda/mu) smoothing barely changes the volume. Each mesh's
    metadata["shared"] marks the vertices it shares with another structure.
    """
    from scipy.sparse import coo_matrix
    iterations = JOINT_SMOOTH_ITERATIONS if iterations is None else iterations
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    keys = list(meshes)
    sizes = [len(meshes[k].vertices) for k in keys]
    allv = np.vstack([meshes[k].vertices for k in keys])
    pairs = cKDTree(allv).query_pairs(tolerance_m, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(allv),) * 2)
    _, group = connected_components(graph, directed=False)
    centre = np.zeros((group.max() + 1, 3))
    np.add.at(centre, group, allv)
    centre /= np.bincount(group)[:, None]
    offsets = np.cumsum([0] + sizes[:-1])
    faces = np.vstack([group[meshes[k].faces + o] for k, o in zip(keys, offsets)])
    smoothed = taubin(centre, faces, iterations)[group]
    owner = np.repeat(np.arange(len(keys)), sizes)
    lo, hi = np.full(len(centre), len(keys)), np.full(len(centre), -1)
    np.minimum.at(lo, group, owner)
    np.maximum.at(hi, group, owner)
    shared = (lo != hi)[group]                               # vertex merged with another structure's
    bits = np.zeros(len(centre), np.int64)
    np.bitwise_or.at(bits, group, np.left_shift(1, owner).astype(np.int64))
    owners = bits[group]                                     # structures (bit = position in keys) at each vertex
    out = {}
    for i, (k, o, n) in enumerate(zip(keys, offsets, sizes)):
        m = trimesh.Trimesh(smoothed[o:o + n], meshes[k].faces, process=False)   # winding unchanged by smoothing
        m.metadata["shared"] = shared[o:o + n]
        m.metadata["neighbours"] = owners[o:o + n] & ~np.int64(1 << i)   # bitmask of the other structures
        out[k] = m
    return out


def taubin(vertices, faces, iterations, lamb=0.5, nu=None):
    """Taubin smoothing (alternating lamb / -nu half-steps, `iterations` half-steps as in trimesh) with an
    equal-weight umbrella operator over each vertex's distinct neighbours.

    trimesh's laplacian_calculation does not normalise rows whose vertex has a zero-length edge (merged
    junction vertices have them): their weights summed to as little as 0.31, which pulled vertices tens of
    mm towards the origin. Here self-loops are dropped and every row sums to 1.
    """
    from scipy.sparse import coo_matrix, diags
    nu = TAUBIN_NU if nu is None else nu
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = e[e[:, 0] != e[:, 1]]
    n = len(vertices)
    W = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    W = ((W + W.T) > 0).astype(float)                       # distinct neighbours, symmetric
    deg = np.asarray(W.sum(1)).ravel()
    L = diags(1.0 / np.maximum(deg, 1)) @ W                  # mean of the neighbours
    v = np.array(vertices, float)
    still = deg == 0                                          # isolated points do not move
    for i in range(iterations):
        step = L @ v - v
        step[still] = 0
        v += (lamb if i % 2 == 0 else -nu) * step
    return v


def printable(mesh, centre):
    """Closed copy for STL / OBJ: centred like the GLB, coincident vertices merged and zero-area faces
    dropped (an STL reader merges coincident vertices itself, which would leave pinched edges)."""
    m = trimesh.Trimesh(mesh.vertices - centre, mesh.faces, process=False)
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    return m


def hidden_faces(mesh, k, order):
    """Faces of see-through structure k that lie on an opaque neighbour or on a lower layer (LAYER):
    they would z-fight with that surface, which shows through instead. mesh.metadata['neighbours'] comes
    from smooth_jointly (bit i = structure order[i])."""
    drop = 0
    for i, j in enumerate(order):
        if j != k and (MODEL_LABELS[j][1][3] >= 1 or LAYER.get(j, 1) < LAYER.get(k, 1)):
            drop |= 1 << i
    common = np.bitwise_and.reduce(mesh.metadata["neighbours"][mesh.faces], axis=1)   # sharing all 3 vertices
    return (common & drop) != 0


def faces_towards(mesh, labels, img, targets, max_mm):
    """Faces whose outward normal meets a label in `targets` within max_mm (sampled every 1 mm)."""
    c = gltf_to_lps(mesh.triangles_center)
    nrm = gltf_to_lps(mesh.face_normals) / 1000.0
    org, sp = np.array(img.GetOrigin()), np.array(img.GetSpacing())
    hit = np.zeros(len(c), bool)
    for d in np.arange(1.0, max_mm + 1e-6, 1.0):
        idx = (((c + nrm * d - org) / sp)[:, ::-1]).T
        hit |= np.isin(ndi.map_coordinates(labels, idx, order=0, mode="constant", cval=0), list(targets))
    return hit


def unit_normals(mesh):
    """Unit vertex normals for every vertex. A vertex whose faces all have (near-)zero area gets the mean
    normal of its neighbours, so the mesh can stay closed and the glTF stays valid."""
    n = np.array(mesh.vertex_normals, float)
    bad = np.linalg.norm(n, axis=1) < 0.5
    neighbours = mesh.vertex_neighbors
    for _ in range(20):
        if not bad.any():
            break
        for v in np.flatnonzero(bad):
            good = [u for u in neighbours[v] if not bad[u]]
            if good:
                n[v] = n[good].sum(0)
                bad[v] = np.linalg.norm(n[v]) < 1e-9
    n[bad] = (0.0, 1.0, 0.0)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def occlusion_fields(occupied, img):
    """Blurred occupancy at each AO_RADII_MM (padded by the largest radius)."""
    pad = int(np.ceil(2 * max(AO_RADII_MM)))
    o = np.pad(occupied.astype(np.float32), pad)
    sp = np.array(img.GetSpacing())[::-1]
    return pad, [ndi.gaussian_filter(o, r / sp) for r in AO_RADII_MM]


def ambient_occlusion(mesh, img, fields, labels, own=None):
    """Per-vertex ambient-occlusion factor (AO_MIN..1): how enclosed the space just outside the surface is,
    from the blurred heart occupancy sampled 1.5 radii out along the normal at each radius.

    Faces shared with another structure (the space 1 mm outside is another label) are internal: hidden
    when the heart is opaque, seen through a transparent wall or a cut. They stay neutral (1) instead of
    reading as fully enclosed.
    """
    pad, blurs = fields
    lps = gltf_to_lps(mesh.vertices)
    nrm = gltf_to_lps(unit_normals(mesh)) / 1000.0
    org, sp = np.array(img.GetOrigin()), np.array(img.GetSpacing())
    occ = []
    for r, blur in zip(AO_RADII_MM, blurs):
        idx = ((lps + nrm * 1.5 * r - org) / sp)[:, ::-1] + pad
        occ.append(ndi.map_coordinates(blur, idx.T, order=1, mode="constant", cval=0.0))
    ao = np.clip(1.0 - AO_STRENGTH * (np.mean(occ, 0) - AO_FLAT), AO_MIN, 1.0)
    outside = ndi.map_coordinates(labels, (((lps + nrm - org) / sp)[:, ::-1]).T, order=0, mode="constant", cval=0)
    ao[(outside != 0) & (outside != (own or 0))] = 1.0
    return ao


def inner_structure(labels, k):
    """True when less than INNER_SHARE of a structure's boundary faces background (it is enclosed, like the LV)."""
    ring = ndi.binary_dilation(labels == k) & (labels != k)
    return (ring & (labels == 0)).sum() < INNER_SHARE * max(ring.sum(), 1)


# ── procedural motion field ──────────────────────────────────────────────
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def rotate_about(v, axis, ang):
    """Rodrigues rotation of vectors v (N,3) about unit axis by ang (N,) rad."""
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return v * c + np.cross(axis, v) * s + axis * (v @ axis)[:, None] * (1 - c)


class CardiacMotion:
    """Displacement fields (glTF metres) for the three morph targets."""

    def __init__(self, axis_lps, aorta_dist_fn,
                 ef=0.60, long_shortening=0.15, twist_deg=10.0,
                 atrial_scale=0.92, a_wave=0.07, aorta_strain=0.08, anchor_mm=ANCHOR_MM,
                 artery_anchor_mm=ARTERY_ANCHOR_MM, wall_mm=WALL_MM):
        self.A = lps_to_gltf(axis_lps["apex"])
        self.B = lps_to_gltf(axis_lps["base"])
        self.C_la = lps_to_gltf(axis_lps["la_center"])
        self.L = np.linalg.norm(self.B - self.A)
        self.u = (self.B - self.A) / self.L
        self.s_anchor = 1.0 + anchor_mm / 1000.0 / self.L    # height (s) where the AV-plane descent has faded out
        self.s_anchor_artery = 1.0 + artery_anchor_mm / 1000.0 / self.L
        self.aorta_dist = aorta_dist_fn
        # radial cavity scale so that radial^2 * longitudinal ~ (1 - EF)
        self.rad = np.sqrt((1 - ef) / (1 - long_shortening))
        self.ls, self.twist = long_shortening, np.radians(twist_deg)
        # atrial systole: the atria contract by 8 % (about a fifth of their volume) and the ventricles' radius grows
        # by 7 % (A-wave filling, about a fifth of the stroke volume), so the blood the atria push out mostly ends
        # up in the ventricles (balanced on the deep-learning chambers, 12 Oct 2026)
        self.atr, self.ao, self.awave = atrial_scale, aorta_strain, a_wave
        self.Rc = 0.022  # nominal LV cavity radius (m), until fit_cavity() has seen the LV
        self.wall = wall_mm / 1000.0
        self.cavity = None   # (s bin centres, cavity radius per (s, angle) bin), from fit_cavity
        self.rv_scale = None  # RV radius around its axis (rv_systole), from motion_context
        e = np.eye(3)[np.argmin(np.abs(self.u))]
        self.e1 = e - (e @ self.u) * self.u
        self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.u, self.e1)

    def _frame(self, p):
        d = p - self.A
        h = d @ self.u                        # height above apex (m)
        s = h / self.L                        # 0 apex .. 1 mitral plane
        r_vec = d - np.outer(h, self.u)
        rho = np.linalg.norm(r_vec, axis=1) + 1e-9
        return h, s, r_vec, rho

    def _angle(self, r_vec):
        return np.arctan2(r_vec @ self.e2, r_vec @ self.e1)    # -pi .. pi around the long axis

    def fit_cavity(self, lv_points, bins=CAVITY_BINS):
        """LV cavity radius per (height, angle) bin: the outermost LV blood-pool point (glTF m) in each bin;
        empty bins take their nearest filled one, then a light smoothing (wrapping round the axis)."""
        _, s, r_vec, rho = self._frame(lv_points)
        ns, nt = bins
        edges = np.linspace(s.min(), s.max() + 1e-9, ns + 1)
        i = np.clip(np.digitize(s, edges) - 1, 0, ns - 1)
        j = ((self._angle(r_vec) + np.pi) / (2 * np.pi) * nt).astype(int) % nt
        grid = np.zeros((ns, nt))
        np.maximum.at(grid, (i, j), rho)
        near = ndi.distance_transform_edt(grid == 0, return_distances=False, return_indices=True)
        grid = grid[tuple(near)]
        grid = ndi.gaussian_filter(grid, 0.7, mode=("nearest", "wrap"))
        self.cavity = ((edges[:-1] + edges[1:]) / 2, grid)

    def cavity_radius(self, s, r_vec):
        if self.cavity is None:
            return np.full_like(s, self.Rc)
        centres, grid = self.cavity
        ns, nt = grid.shape
        fi = np.interp(s, centres, np.arange(ns))                                    # clamped to the end bins
        fj = (self._angle(r_vec) + np.pi) / (2 * np.pi) * nt - 0.5 + 1             # +1: the wrap padding below
        padded = np.concatenate([grid[:, -1:], grid, grid[:, :1]], 1)
        return ndi.map_coordinates(padded, [fi, fj], order=1, mode="nearest")

    def rv_systole(self, p, c_rv, ef=0.45, long_shortening=0.20):
        """RV contraction. The free wall moves towards an axis through the RV centre (parallel to the LV
        axis), with more base-to-apex shortening than the LV (TAPSE). The septal side follows the LV."""
        h, s, r_vec, rho = self._frame(p)
        w_v = 1 - smoothstep(0.95, 1.20, s)
        taper = smoothstep(-0.05, 0.25, s)
        d = p - c_rv; d_perp = d - np.outer(d @ self.u, self.u)
        rad = np.sqrt((1 - ef) / (1 - long_shortening))
        rv = -d_perp * ((1 - rad) * taper * w_v)[:, None] - np.outer(long_shortening * np.clip(h, 0, None) * w_v, self.u)
        out = c_rv - self.A; out = out - (out @ self.u) * self.u; out /= np.linalg.norm(out) + 1e-9   # LV axis -> RV
        scale = self.rv_scale or np.linalg.norm(d_perp, axis=1).max()
        free = smoothstep(-0.6, -0.1, (d_perp @ out) / (scale + 1e-9))   # 0 only on the septal side
        return free[:, None] * rv + (1 - free)[:, None] * self.ventricular_systole(p)

    def rv_atrial_systole(self, p, c_rv, gain=None, lengthening=0.03):
        """RV filling in atrial systole (A wave): the free wall moves out from the RV axis, the base up; the septal
        side follows the LV's atrial_systole."""
        h, s, r_vec, rho = self._frame(p)
        above = smoothstep(0.95, 1.15, s)
        w_v = 1 - above
        d = p - c_rv; d_perp = d - np.outer(d @ self.u, self.u)
        gain = self.awave if gain is None else gain
        rv = d_perp * (gain * smoothstep(0.0, 0.3, s) * w_v)[:, None] + np.outer(lengthening * np.clip(h, 0, None) * w_v, self.u)
        out = c_rv - self.A; out = out - (out @ self.u) * self.u; out /= np.linalg.norm(out) + 1e-9
        scale = self.rv_scale or np.linalg.norm(d_perp, axis=1).max()
        free = smoothstep(-0.6, -0.1, (d_perp @ out) / (scale + 1e-9))
        return free[:, None] * rv + (1 - free)[:, None] * self.atrial_systole(p)

    def ventricular_systole(self, p, s_anchor=None):
        h, s, r_vec, rho = self._frame(p)
        w_v = 1 - smoothstep(0.95, 1.20, s)   # ventricle zone (fades above AV plane)
        # radial: the cavity (inside the LV's own radius at this height and angle) scales down; the wall keeps
        # its volume (myocardium is incompressible), so it thickens as it shortens; beyond the wall, the
        # cross-section area outside is kept
        taper = smoothstep(-0.05, 0.25, s)    # apex barely moves radially
        Rc = self.cavity_radius(s, r_vec)
        f = 1 - (1 - self.rad) * taper * w_v                  # cavity scale
        g = 1 / (1 - self.ls * w_v)                           # wall area gain that offsets the shortening
        wall = np.clip(rho ** 2 - Rc ** 2, 0, (Rc + self.wall) ** 2 - Rc ** 2)
        rho_new = np.where(rho < Rc, rho * f,
                           np.sqrt(np.maximum(rho ** 2 + Rc ** 2 * (f ** 2 - 1) + wall * (g - 1), 0)))
        d_rad = (r_vec / rho[:, None]) * (rho_new - rho)[:, None]
        # longitudinal: base descends toward (nearly fixed) apex
        d_long = -np.outer(self.ls * np.clip(h, 0, None) * w_v, self.u)
        # the atria's floor descends with the AV plane; their roof is held by the veins (anchor), so they stretch
        above = smoothstep(0.95, 1.20, s)
        hold = 1 - smoothstep(1.0, self.s_anchor if s_anchor is None else s_anchor, s)
        d_long += -np.outer(self.ls * self.L * above * hold, self.u)
        # twist: apex counter-clockwise vs base clockwise (viewed from apex)
        ang = self.twist * (0.5 - np.clip(s, 0, 1)) * w_v
        d_tw = rotate_about(r_vec, self.u, ang) - r_vec
        return d_rad + d_long + d_tw

    def atrial_systole(self, p, center=None):
        h, s, r_vec, rho = self._frame(p)
        above = smoothstep(0.95, 1.15, s)
        d_atr = (p - (self.C_la if center is None else center)) * (self.atr - 1) * above[:, None]
        # late (A-wave) filling: the cavity widens a little and the AV plane moves up toward the atria; the wall
        # keeps its volume (thins as it lengthens), as in ventricular_systole
        w_v = 1 - above
        e = self.awave * smoothstep(0.0, 0.3, s) * w_v                 # cavity radius gain
        g = 1 / (1 + 0.03 * w_v)                                       # wall area factor offsetting the lengthening
        Rc = self.cavity_radius(s, r_vec)
        wall = np.clip(rho ** 2 - Rc ** 2, 0, (Rc + self.wall) ** 2 - Rc ** 2)
        rho_new = np.where(rho < Rc, rho * (1 + e),
                           np.sqrt(np.maximum(rho ** 2 + Rc ** 2 * ((1 + e) ** 2 - 1) + wall * (g - 1), 0)))
        d_v = (r_vec / rho[:, None]) * (rho_new - rho)[:, None]
        d_v += np.outer(0.03 * np.clip(h, 0, None) * w_v, self.u)
        return d_atr + d_v

    def aortic_distension(self, p, aorta_center_fn, dist_fn=None):
        """Great artery expands during ejection (the aorta; the pulmonary artery when given its own fields)."""
        dist = (self.aorta_dist if dist_fn is None else dist_fn)(p)   # m outside the vessel surface
        w = np.exp(-np.clip(dist, 0, None) / 0.004)   # vessel + adjacent wall
        c = aorta_center_fn(p)
        rad = p - c
        rad[:, 1] = 0                                 # expand in-plane (tube ~ vertical)
        return rad * self.ao * w[:, None]


def cycle_weights(n=40):
    """Morph weights over one R-R interval (phase 0..1), rest pose = diastasis."""
    ph = np.linspace(0, 1, n + 1)
    # ventricular: IVC 0-0.05, ejection 0.05-0.33, IVR 0.33-0.42, rapid filling 0.42-0.60
    wv = np.interp(ph, [0, .05, .20, .33, .42, .50, .60, .80, 1.0],
                       [0, .05, .75, 1.0, .95, .45, .10, 0.0, 0.0])
    # atrial kick: P-wave at ~0.82-0.98
    wa = np.interp(ph, [0, .80, .88, .93, .98, 1.0], [0, 0, .7, 1, .15, 0])
    # aortic distension follows ejection with a short lag
    wo = np.interp(ph, [0, .08, .22, .35, .50, .80, 1.0], [0, .1, 1, .7, .35, .1, 0])
    return ph, np.stack([wv, wa, wo], 1)


def motion_targets(motion, k, p, ra_centre, rv_centre, aorta_center, pa_center, pa_dist, dao=None):
    """The three morph targets (TARGETS) of structure k's own motion at points p. The model blends them
    between neighbours (motion_context's 'targets'). dao: (centre, dist) fields of the descending aorta."""
    zero = np.zeros_like(p)
    if k == 11:         # descending aorta: fixed to the spine, does not follow the valve plane; distends
        return [zero, zero, motion.aortic_distension(p, *dao) if dao else zero]
    if k == 9:          # IVC: held by the diaphragm and liver; it moves only where it blends into the RA
        return [zero, zero, zero]
    if k == 10:         # pulmonary veins: ride with the LA, at a third of its contraction
        return [motion.ventricular_systole(p), 0.3 * motion.atrial_systole(p), zero]
    if k in (3, 7):     # great arteries: the root follows the valve plane, the arch holds the top; they distend
        centre, dist = (pa_center, pa_dist) if k == 7 else (aorta_center, None)
        return [motion.ventricular_systole(p, s_anchor=motion.s_anchor_artery), zero,
                motion.aortic_distension(p, centre, dist)]
    if k == 8:          # SVC: rides with the RA, at a third of its contraction
        return [motion.ventricular_systole(p), 0.3 * motion.atrial_systole(p, center=ra_centre),
                motion.aortic_distension(p, aorta_center)]
    if k == 6 and rv_centre is not None:
        return [motion.rv_systole(p, rv_centre), motion.rv_atrial_systole(p, rv_centre), motion.aortic_distension(p, aorta_center)]
    atrium = ra_centre if k == 5 else None
    return [motion.ventricular_systole(p), motion.atrial_systole(p, center=atrium),
            motion.aortic_distension(p, aorta_center)]


# ── glTF writer (morph targets + weight animation) ────────────────────────
class GLTFBuilder:
    def __init__(self):
        self.bin = bytearray()
        self.g = {"asset": {"version": "2.0", "generator": "UTM 3DHeartModelV1"},
                  "scene": 0, "scenes": [{"name": "Heart", "nodes": []}], "nodes": [],
                  "meshes": [], "materials": [], "accessors": [], "bufferViews": [],
                  "buffers": [], "animations": []}

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin); self.bin += data
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.g["bufferViews"].append(bv)
        return len(self.g["bufferViews"]) - 1

    def accessor(self, arr, kind, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        ctype = {np.dtype("float32"): 5126, np.dtype("uint32"): 5125, np.dtype("uint8"): 5121}[arr.dtype]
        acc = {"bufferView": self._view(arr.tobytes(), target), "componentType": ctype,
               "count": int(arr.shape[0]), "type": kind}
        if arr.dtype == np.uint8:
            acc["normalized"] = True                     # 0..255 -> 0..1 (vertex colours)
        if minmax:
            a2 = arr.reshape(arr.shape[0], -1)
            acc["min"] = a2.min(0).tolist(); acc["max"] = a2.max(0).tolist()
        self.g["accessors"].append(acc)
        return len(self.g["accessors"]) - 1

    def add_mesh(self, name, mesh, rgba, targets, target_names, shade=None):
        """shade: optional per-vertex factor 0..1 (baked ambient occlusion), stored as COLOR_0, which glTF
        viewers multiply with the base colour."""
        v = mesh.vertices.astype(np.float32)
        n = unit_normals(mesh).astype(np.float32)
        f = mesh.faces.astype(np.uint32).reshape(-1)
        blend = rgba[3] < 1
        self.g["materials"].append({
            "name": name, "doubleSided": True, "alphaMode": "BLEND" if blend else "OPAQUE",
            "pbrMetallicRoughness": {"baseColorFactor": list(rgba), "metallicFactor": 0.0,
                                     "roughnessFactor": 0.55}})
        attributes = {"POSITION": self.accessor(v, "VEC3", 34962, True),
                      "NORMAL": self.accessor(n, "VEC3", 34962)}
        if shade is not None:
            g = np.round(np.clip(shade, 0, 1) * 255).astype(np.uint8)
            attributes["COLOR_0"] = self.accessor(np.stack([g, g, g, np.full_like(g, 255)], 1), "VEC4", 34962)
        prim = {"attributes": attributes,
                "indices": self.accessor(f, "SCALAR", 34963), "mode": 4,
                "material": len(self.g["materials"]) - 1,
                "targets": [{"POSITION": self.accessor(t.astype(np.float32), "VEC3", 34962, True)}
                            for t in targets]}
        self.g["meshes"].append({"name": name, "primitives": [prim],
                                 "weights": [0.0] * len(targets),
                                 "extras": {"targetNames": target_names}})
        self.g["nodes"].append({"name": name, "mesh": len(self.g["meshes"]) - 1})
        self.g["scenes"][0]["nodes"].append(len(self.g["nodes"]) - 1)
        return len(self.g["nodes"]) - 1

    def add_weight_animation(self, name, node_ids, times, weights):
        t_acc = self.accessor(times.astype(np.float32), "SCALAR", minmax=True)
        w_acc = self.accessor(weights.astype(np.float32).reshape(-1), "SCALAR")
        samplers, channels = [], []
        for nid in node_ids:
            samplers.append({"input": t_acc, "output": w_acc, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": nid, "path": "weights"}})
        self.g["animations"].append({"name": name, "samplers": samplers, "channels": channels})

    def write_glb(self, path):
        g = dict(self.g); g["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(g, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        binb = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        total = 12 + 8 + len(js) + 8 + len(binb)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<III", 0x46546C67, 2, total))
            fh.write(struct.pack("<II", len(js), 0x4E4F534A)); fh.write(js)
            fh.write(struct.pack("<II", len(binb), 0x004E4942)); fh.write(binb)

    def write_gltf(self, path):
        binname = os.path.splitext(os.path.basename(path))[0] + ".bin"
        g = dict(self.g); g["buffers"] = [{"byteLength": len(self.bin), "uri": binname}]
        with open(os.path.join(os.path.dirname(path), binname), "wb") as fh:
            fh.write(self.bin)
        with open(path, "w") as fh:
            json.dump(g, fh, indent=1)


# ── main ─────────────────────────────────────────────────────────────────
def dl_chambers_available(lab_img):
    """True when segment_heart_dl.py has written chamber labels (licensed task) on the heart_labels grid."""
    path, stats = os.path.join(hio.OUT_MODELS, "heart_labels_dl.nii.gz"), os.path.join(hio.OUT_MODELS, "dl_segmentation.json")
    if not (os.path.exists(path) and os.path.exists(stats)):
        return False
    with open(stats) as fh:
        if not json.load(fh).get("chambers"):
            return False
    img = sitk.ReadImage(path)
    return img.GetSize() == lab_img.GetSize() and np.allclose(img.GetOrigin(), lab_img.GetOrigin())


def chamber_source(lab_img, chambers="auto"):
    """'dl' (TotalSegmentator heartchambers_highres, segment_heart_dl.py) or 'rules' (segment_heart.py)."""
    if chambers not in ("auto", "dl", "rules"):
        raise ValueError(f"chambers must be auto, dl or rules, not {chambers}")
    available = dl_chambers_available(lab_img)
    if chambers == "dl" and not available:
        raise FileNotFoundError("no deep-learning chambers: run segment_heart_dl.py with a TotalSegmentator licence")
    return "dl" if chambers == "dl" or chambers == "auto" and available else "rules"


def lv_long_axis(labels, img):
    """LV long axis in patient LPS mm from the LV (1) and LA (2) labels: base = centre of the LV next to the LA
    (the mitral valve), apex = the LV farthest from it, la_center = the LA's centre."""
    sp = np.array(img.GetSpacing())[::-1]
    lv, la = labels == 1, labels == 2
    to_lps = lambda idx: [float(v) for v in img.TransformContinuousIndexToPhysicalPoint([float(i) for i in idx[::-1]])]
    mitral = lv & (ndi.distance_transform_edt(~la, sampling=sp) <= 2.0)
    base = np.argwhere(mitral).mean(0)
    pts = np.argwhere(lv)
    far = np.linalg.norm((pts - base) * sp, axis=1)
    apex = pts[far >= np.percentile(far, 99.5)].mean(0)
    return {"apex": to_lps(apex), "base": to_lps(base), "la_center": to_lps(np.argwhere(la).mean(0))}


def model_labels(lab_img, log=print, chambers="auto"):
    """The model's labels: the chambers (deep learning when available, else the rules; chamber_source) with the
    TotalSegmentator vessels merged in, the coronary arteries (COR_LABEL, on background only) and the fat display
    layer (EAT_LABEL), when their inputs exist. Also returns the vessel map and the mapped EAT mask (or None)."""
    deep = chamber_source(lab_img, chambers) == "dl"
    labels = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(hio.OUT_MODELS, "heart_labels_dl.nii.gz")) if deep else lab_img)
    if deep:
        log("  chambers: deep learning (TotalSegmentator heartchambers_highres, heart_labels_dl.nii.gz)")
    ves = vs.load(lab_img, log)
    if ves is not None:
        labels = vs.merge(labels, ves, lab_img.GetSpacing()[::-1], deep=deep)
        log("  merged the TotalSegmentator vessels (vessels_dl.nii.gz)")
    cor = coronaries.load()
    if cor is not None:
        on = coronary_grid(cor, lab_img) & (labels == 0)
        labels = labels.copy()
        labels[on] = COR_LABEL
        log(f"  coronary arteries (coronaries_dl.nii.gz): {on.sum() * np.prod(lab_img.GetSpacing()) / 1000:.1f} mL on the grid")
    eat = load_eat(lab_img, log)
    if eat is not None:
        ct = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(hio.OUT_MODELS, "heart_ct_1mm.nii.gz")))
        fat = fat_layer(eat, ct, labels, lab_img.GetSpacing()[::-1])
        labels = labels.copy()
        labels[fat] = EAT_LABEL
        log(f"  epicardial fat display layer: {fat.sum() * np.prod(lab_img.GetSpacing()) / 1000:.1f} mL of CCTA fat")
    return labels, ves, eat


def motion_blend(labels, spacing_zyx, sigma_mm=MOTION_SIGMA_MM):
    """Weights for blending the structures' own motion: each structure's indicator smoothed by sigma_mm, on a
    crop of the grid. The fat and the coronary arteries (CARRIED) have no motion of their own: their voxels count
    for the nearest structure, so they move with what they lie on. Returns (structure ids, (K, z, y, x) float32
    stack, crop offset)."""
    sp = np.asarray(spacing_zyx, float)
    carrier = labels.copy()
    fat = np.isin(labels, CARRIED)
    if fat.any():
        near = ndi.distance_transform_edt((labels == 0) | fat, sampling=sp, return_distances=False,
                                          return_indices=True)
        carrier[fat] = labels[tuple(i[fat] for i in near)]
    pad = np.ceil(4 * sigma_mm / sp).astype(int) + 1
    nz = np.argwhere(carrier > 0)
    lo, hi = np.maximum(nz.min(0) - pad, 0), np.minimum(nz.max(0) + pad + 1, labels.shape)
    crop = carrier[tuple(slice(a, b) for a, b in zip(lo, hi))]
    ids = [int(k) for k in np.unique(crop) if k > 0]
    return ids, np.stack([ndi.gaussian_filter((crop == k).astype(np.float32), sigma_mm / sp) for k in ids]), lo


def motion_context(lab_img, labels, axis):
    """Everything the morph targets need besides the points: the CardiacMotion, chamber centres and
    great-artery fields (glTF space), the whole-heart centroid the model is centred on, and 'targets':
    the blended morph targets at any points (shared surfaces get the same motion from both sides)."""
    sp = np.array(lab_img.GetSpacing()); org = np.array(lab_img.GetOrigin())

    def gltf_to_index(p):
        lps = np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1) * 1000.0
        return ((lps - org) / sp)[:, ::-1]  # z,y,x

    def vessel_fields(vessel):
        """glTF-space helpers for a great artery: distance to its surface, and its centre at each height."""
        edt = ndi.distance_transform_edt(~vessel, sampling=sp[::-1]) / 1000.0
        zk = np.array([z for z in range(vessel.shape[0]) if vessel[z].any()])
        yxk = np.array([np.argwhere(vessel[z]).mean(0) for z in zk])

        def dist(p):
            return ndi.map_coordinates(edt, gltf_to_index(p).T, order=1, mode="nearest")

        def centre(p):
            idx = gltf_to_index(p)
            y = np.interp(idx[:, 0], zk, yxk[:, 0]); x = np.interp(idx[:, 0], zk, yxk[:, 1])
            return lps_to_gltf(np.stack([x, y, idx[:, 0]], 1) * sp + org)   # x,y,z index -> LPS -> glTF
        return dist, centre

    aorta_dist, aorta_center = vessel_fields(labels == 3)
    pa_dist, pa_center = vessel_fields(labels == 7) if (labels == 7).any() else (aorta_dist, aorta_center)
    to_gltf = lambda idx: lps_to_gltf(np.array(lab_img.TransformContinuousIndexToPhysicalPoint(idx[::-1].tolist())))
    ra_centre = to_gltf(np.argwhere(labels == 5).mean(0)) if (labels == 5).any() else None
    rv_centre = to_gltf(np.argwhere(labels == 6).mean(0)) if (labels == 6).any() else None

    motion = CardiacMotion(axis, aorta_dist)
    voxels_gltf = lambda m: lps_to_gltf(np.argwhere(m)[:, ::-1] * sp + org)
    if (labels == 1).any():
        motion.fit_cavity(voxels_gltf(labels == 1))
    if rv_centre is not None:
        d = voxels_gltf(labels == 6) - rv_centre
        motion.rv_scale = float(np.linalg.norm(d - np.outer(d @ motion.u, motion.u), axis=1).max())

    # centre the model on the whole-heart centroid
    cen = to_gltf(np.argwhere(labels > 0).mean(0))
    dao = vessel_fields(labels == 11)[::-1] if (labels == 11).any() else None    # (centre, dist)
    ids, blend, lo = motion_blend(labels, sp[::-1])

    def targets(p):
        """Morph targets at points p (glTF m): sum over structures k of w_k(p) * k's own targets, where w is
        motion_blend normalised to 1. They depend on the position only, so a vertex shared by two structures
        gets the same motion in both meshes."""
        idx = (gltf_to_index(p) - lo).T
        w = np.stack([ndi.map_coordinates(b, idx, order=1, mode="constant") for b in blend], 1)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        w[w < 1e-3] = 0
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        out = [np.zeros_like(p) for _ in TARGETS]
        for j, k in enumerate(ids):
            sel = w[:, j] > 0
            if sel.any():
                T = motion_targets(motion, k, p[sel], ra_centre, rv_centre, aorta_center, pa_center, pa_dist, dao)
                for o, t in zip(out, T):
                    o[sel] += w[sel, j, None] * t
        return out
    return dict(motion=motion, cen=cen, ra_centre=ra_centre, rv_centre=rv_centre, dao=dao,
                aorta_center=aorta_center, pa_center=pa_center, pa_dist=pa_dist, targets=targets)


def main(bpm=72.0, total_faces=450000, log=print, chambers="auto"):
    t0 = time.time()
    lab_img = sitk.ReadImage(os.path.join(hio.OUT_MODELS, "heart_labels.nii.gz"))
    with open(os.path.join(hio.OUT_MODELS, "segmentation_stats.json")) as fh:
        stats = json.load(fh)
    source = chamber_source(lab_img, chambers)
    labels, ves, eat = model_labels(lab_img, log, source)
    axis = lv_long_axis(labels, lab_img) if source == "dl" else stats["lv_long_axis_lps_mm"]
    ctx = motion_context(lab_img, labels, axis)
    cen = ctx["cen"]

    order = [k for k in (4, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, COR_LABEL, EAT_LABEL) if (labels == k).any()]
    joint = [k for k in order if k != COR_LABEL]           # meshed together on the 1 mm grid
    gb = GLTFBuilder()
    node_ids, meshes, morphs = [], {}, {}
    ph, W = cycle_weights()
    period = 60.0 / bpm
    log(f"  meshing {len(joint)} structures on one shared surface field ...")
    surfaces = label_meshes(np.where(labels == COR_LABEL, 0, labels), lab_img, joint, total_faces)
    if COR_LABEL in order:                                  # on their own, at the CT's native resolution
        surfaces[COR_LABEL] = coronary_mesh(coronaries.load(), COR_FACE_SHARE * total_faces)
    heart_ao = occlusion_fields(labels > 0, lab_img)
    closed = {MODEL_LABELS[k][0]: surfaces[k].copy() for k in order}       # STL / OBJ (3D printing) need closed shells
    for k in order:
        if MODEL_LABELS[k][1][3] < 1:
            # a see-through structure drops the faces it shares with an opaque neighbour or a lower layer
            # (they lie exactly on that surface and would z-fight; that surface shows through instead).
            # The wall keeps the faces under the fat, so there is no hole in it there.
            m = surfaces[k]
            keep = ~hidden_faces(m, k, joint)
            if k == EAT_LABEL:
                # the fat sits 1-2 mm off the 110 HU heart outline, so its inner side is not a shared surface;
                # it would shade as deep crevices through the translucent fat. A translucent coating only
                # needs its outer side: drop faces whose normal meets the heart within FAT_TOUCH_MM
                keep &= ~faces_towards(m, labels, lab_img, set(order) - {EAT_LABEL}, FAT_TOUCH_MM)
            m.update_faces(keep)
            m.update_faces(m.nondegenerate_faces())        # open anyway; zero-area faces only add validator notes
            m.remove_unreferenced_vertices()
    parts = [(k, MODEL_LABELS[k][0], MODEL_LABELS[k][1], surfaces[k],
              occlusion_fields(labels == k, lab_img) if k in joint and inner_structure(labels, k) else heart_ao)
             for k in order]
    for k, name, rgba, m, ao_fields in parts:
        p = m.vertices
        shade = ambient_occlusion(m, lab_img, ao_fields, labels, k)
        if k == EAT_LABEL:
            shade = 1.0 - EAT_AO_SCALE * (1.0 - shade)
        T = ctx["targets"](p)
        m.vertices = p - cen
        meshes[name] = m
        morphs[name] = T
        nid = gb.add_mesh(name, m, rgba, T, TARGETS, shade)
        node_ids.append(nid)
        log(f"    {name}: {len(m.vertices)} verts, {len(m.faces)} faces, "
            f"{abs(m.volume) * 1e6 if m.is_watertight else float('nan'):.1f} mL, AO {shade.min():.2f}-{shade.max():.2f}, max disp "
            f"{max(np.linalg.norm(t, axis=1).max() for t in T) * 1000:.1f} mm")

    gb.add_weight_animation("CardiacCycle", node_ids, ph * period, W)
    gb.g["extras"] = {"bpm": bpm, "period_s": period, "rest_phase": "75% R-R (diastasis)",
                      "source_series": stats["series"], "motion": "procedural morph targets",
                      "volumes_ml": stats["labels"],
                      "chambers": "deep learning (TotalSegmentator heartchambers_highres)" if source == "dl" else "rule-based (segment_heart.py)",
                      # the patient LPS point (mm) at the model origin: model + origin = lps_to_gltf(LPS)
                      "origin_lps_mm": [round(float(v), 4) for v in gltf_to_lps(cen)]}
    vox_ml = float(np.prod(lab_img.GetSpacing())) / 1000.0
    gb.g["extras"]["model_volumes_ml"] = {MODEL_LABELS[k][0]: round(float((labels == k).sum() * vox_ml), 1) for k in order}
    if ves is not None and os.path.exists(vs.stats_file()):
        with open(vs.stats_file()) as fh:
            v = json.load(fh)
        gb.g["extras"]["vessels"] = {"source": f"TotalSegmentator {v['totalsegmentator']} ({v['task']} task)",
                                     "structures": [MODEL_LABELS[k][0] for k in (8, 9, 10, 11) if k in order]}
    if COR_LABEL in order and os.path.exists(coronaries.stats_file()):
        with open(coronaries.stats_file()) as fh:
            c = json.load(fh)
        gb.g["extras"]["coronaries"] = {"source": f"TotalSegmentator {c['totalsegmentator']} ({c['task']} task)",
                                        "volume_ml": c["volume_ml"], "pieces": c["pieces"],
                                        "mask_grid_mm": c["grid"]["spacing_mm"]}
    eat_stats = os.path.join(hio.OUT_MODELS, "eat_stats.json")
    if eat is not None and os.path.exists(eat_stats):
        with open(eat_stats) as fh:
            e = json.load(fh)
        gb.g["extras"]["eat"] = {"volume_ml": e["eat_volume_ml"], "mean_hu": e["eat_mean_hu"], "series": e["series"]}

    out = hio.OUT_MODELS
    glb = os.path.join(out, "heart_animated.glb")
    gb.write_glb(glb); gb.write_gltf(os.path.join(out, "heart_animated.gltf"))
    for name, m in meshes.items():
        slug = MODEL_SLUGS[next(k for k, v in MODEL_LABELS.items() if v[0] == name)]
        solid = printable(closed[name], cen) if name in closed else m
        solid.export(os.path.join(out, f"heart_{slug}.stl"))
        solid.export(os.path.join(out, f"heart_{slug}.obj"))
    write_viewer_html(os.path.join(out, "heart_viewer.html"), "heart_animated.glb")

    # volume curve of LV over the cycle, with its morph targets as in the GLB (sanity check of the procedural EF)
    lv, T = meshes["LV blood pool"], morphs["LV blood pool"]
    vols = []
    for w in W:
        mm = trimesh.Trimesh(lv.vertices + sum(wi * t for wi, t in zip(w, T)), lv.faces, process=False)
        vols.append(abs(mm.volume) * 1e6)
    info = {"bpm": bpm, "period_s": period, "phase": ph.tolist(), "weights": W.tolist(),
            "lv_volume_ml": vols, "lv_ef_percent": 100 * (max(vols) - min(vols)) / max(vols),
            "faces": {n: int(len(m.faces)) for n, m in meshes.items()},
            "glb_mb": os.path.getsize(glb) / 1e6}
    with open(os.path.join(out, "animation_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    save_cycle_plot(info, os.path.join(hio.OUT_VIS, "cardiac_cycle.png"))
    log(f"Done in {time.time() - t0:.1f}s -> {glb} ({info['glb_mb']:.1f} MB), "
        f"LV EDV {max(vols):.1f} mL, ESV {min(vols):.1f} mL, EF {info['lv_ef_percent']:.0f}%")
    return info


def save_cycle_plot(info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ph = np.array(info["phase"]) * 100; W = np.array(info["weights"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(ph, info["lv_volume_ml"], color="#c0392b", lw=2)
    a1.set_ylabel("LV blood-pool volume (mL)")
    a1.set_title(f"Procedural cardiac cycle @ {info['bpm']:.0f} bpm  -  LV EF {info['lv_ef_percent']:.0f}%")
    for x, t in [(5, "AVO"), (33, "AVC"), (42, "MVO"), (98, "MVC")]:
        a1.axvline(x, color="#999", ls=":"); a1.text(x, a1.get_ylim()[1], t, fontsize=8, va="top")
    a1.axvline(75, color="#2980b9", ls="--"); a1.text(75, a1.get_ylim()[0], " CT phase 75%", color="#2980b9", fontsize=8)
    for i, (n, c) in enumerate([("ventricular systole", "#c0392b"), ("atrial systole", "#2980b9"),
                                ("great-artery distension", "#f39c12")]):
        a2.plot(ph, W[:, i], label=n, color=c, lw=2)
    a2.set_ylabel("morph weight"); a2.set_xlabel("% R-R interval"); a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close(fig)


VIEWER_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>3D Heart Model</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#0e1117;--panel:rgba(20,24,33,.85);--fg:#e8e8e8;--muted:#9aa0a6;--acc:#e0474c}
 html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}
 #c{width:100%;height:100%;display:block}
 #ui{position:absolute;top:10px;left:10px;background:var(--panel);padding:10px 12px;border-radius:8px;min-width:210px}
 summary{cursor:pointer;margin-bottom:4px} #ui label{display:flex;align-items:center;gap:6px;margin:3px 0}
 .sw{width:10px;height:10px;border-radius:2px;display:inline-block}
 #ui input[type=range]{width:100%} button{background:var(--acc);color:#fff;border:0;border-radius:5px;padding:4px 10px;cursor:pointer}
 #ecg{position:absolute;bottom:10px;left:10px;width:calc(100% - 20px);height:60px;background:var(--panel);border-radius:8px}
 .muted{color:var(--muted);font-size:11px} #err{position:absolute;top:50%;width:100%;text-align:center}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<canvas id="c"></canvas>
<div id="ui"><details id="dt" open><summary><b style="color:var(--acc)">&#x1FAC0; 3D Heart &mdash; cardiac cycle</b></summary>
 <div id="parts"></div>
 <label>Heart rate <b id="bpmv"></b> bpm</label><input id="bpm" type="range" min="30" max="150" value="__BPM__">
 <label>Wall opacity</label><input id="op" type="range" min="0" max="100" value="45">
 <div style="margin-top:6px;display:flex;gap:6px"><button id="pp">Pause</button><button id="rs">Reset view</button></div>
 <div class="muted" id="ph" style="margin-top:6px"></div>
 <div class="muted">Patient CCTA &middot; procedural motion &middot; drag to rotate</div></details></div>
<canvas id="ecg"></canvas><div id="err"></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
if(innerWidth<640) document.getElementById('dt').open=false;
const SRC="__SRC__", BASE_BPM=__BPM__;
const cv=document.getElementById('c'), R=new THREE.WebGLRenderer({canvas:cv,antialias:true});
R.setPixelRatio(devicePixelRatio); const S=new THREE.Scene(); S.background=new THREE.Color(0x0e1117);
const C=new THREE.PerspectiveCamera(35,1,0.001,10); C.position.set(0,0.02,0.33);
const O=new OrbitControls(C,cv); O.enableDamping=true;
S.add(new THREE.HemisphereLight(0xffffff,0x332222,1.6)); const dl=new THREE.DirectionalLight(0xffffff,2.2); dl.position.set(.3,.5,.6); C.add(dl); S.add(C);
function rs(){const w=innerWidth||800,h=innerHeight||600;R.setSize(w,h,false);C.aspect=w/h;C.updateProjectionMatrix();}
addEventListener('resize',rs); rs();
function fit(obj){const b=new THREE.Box3().setFromObject(obj), sz=b.getSize(new THREE.Vector3()), c=b.getCenter(new THREE.Vector3());
  const r=sz.length()/2, f=THREE.MathUtils.degToRad(C.fov)/2, fh=Math.atan(Math.tan(f)*Math.max(C.aspect||1,0.6));
  const d=1.15*r/Math.sin(Math.min(f,fh)); O.target.copy(c); C.position.set(c.x+d*0.15,c.y+d*0.1,c.z+d); C.near=d/100; C.far=d*10; C.updateProjectionMatrix(); O.update(); O.saveState();}
let mixer,action,playing=true,clip; const parts=document.getElementById('parts');
new GLTFLoader().load(SRC,g=>{
  S.add(g.scene); window.model=g.scene; fit(g.scene); mixer=new THREE.AnimationMixer(g.scene); clip=g.animations[0]; action=mixer.clipAction(clip); action.play();
  g.scene.traverse(o=>{ if(!o.isMesh) return; const m=o.material; m.depthWrite=!m.transparent; if(m.transparent) o.renderOrder=2;
    const col='#'+m.color.getHexString(); const l=document.createElement('label');
    l.innerHTML=`<input type=checkbox checked><span class=sw style="background:${col}"></span>${o.name.replace(/_/g,' ')}`;
    l.firstChild.onchange=e=>o.visible=e.target.checked; parts.appendChild(l); if(m.transparent&&/^Myocardium/.test(o.name)) window.wall=m; });
  setBpm();
},undefined,e=>document.getElementById('err').textContent='Could not load '+SRC+' ('+e.message+'). Serve this folder over http (python -m http.server).');
const bpm=document.getElementById('bpm'), bpmv=document.getElementById('bpmv');
function setBpm(){bpmv.textContent=bpm.value; if(action) action.timeScale=bpm.value/BASE_BPM;}
bpm.oninput=setBpm; document.getElementById('op').oninput=e=>{if(window.wall) window.wall.opacity=e.target.value/100;};
document.getElementById('pp').onclick=e=>{playing=!playing; e.target.textContent=playing?'Pause':'Play';};
document.getElementById('rs').onclick=()=>{O.reset();};
const ecg=document.getElementById('ecg'), ex=ecg.getContext('2d'); const hist=[];
function ecgv(p){ // toy ECG in phase units (R at 0)
  const g=(c,w,a)=>a*Math.exp(-(((p-c)/w)**2)); return g(0,.012,1)+g(-.02,.01,-.15)+g(.025,.012,-.25)+g(.28,.05,.3)+g(.87,.03,.15)+g(1,.012,1);}
const clock=new THREE.Clock();
let lastW=0;
(function loop(){requestAnimationFrame(loop); if(innerWidth!==lastW){lastW=innerWidth; rs(); if(window.model) fit(window.model);} const dt=clock.getDelta(); if(mixer&&playing) mixer.update(dt);
  O.update(); R.render(S,C);
  if(action){const p=(action.time%clip.duration)/clip.duration; const names=[[.05,'Isovolumic contraction'],[.33,'Ejection (systole)'],[.42,'Isovolumic relaxation'],[.60,'Rapid filling'],[.80,'Diastasis (CT rest pose)'],[.98,'Atrial systole'],[1.01,'Isovolumic contraction']];
    document.getElementById('ph').textContent='Phase '+(p*100).toFixed(0)+'% R-R — '+names.find(n=>p<n[0])[1];
    const w=ecg.clientWidth,h=ecg.clientHeight; if(ecg.width!==w){ecg.width=w;ecg.height=h;}
    ex.clearRect(0,0,w,h); ex.strokeStyle='#e0474c'; ex.lineWidth=2; ex.beginPath();
    for(let i=0;i<w;i++){const q=p-(w-1-i)/w*2; const pp=((q%1)+1)%1; const y=h*0.75-ecgv(pp)*h*0.6; i?ex.lineTo(i,y):ex.moveTo(i,y);} ex.stroke();
    ex.fillStyle='#fff'; ex.fillRect(w-3,0,2,h);}
})();
</script></body></html>"""


def write_viewer_html(path, src, bpm=72):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(VIEWER_HTML.replace("__SRC__", src).replace("__BPM__", str(int(bpm))))


def viewer_html_inline(glb_path, bpm=72):
    """Viewer HTML with the GLB embedded as a data URI (for Streamlit)."""
    with open(glb_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return VIEWER_HTML.replace("__SRC__", "data:model/gltf-binary;base64," + b64).replace("__BPM__", str(int(bpm)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=72)
    ap.add_argument("--faces", type=int, default=450000)
    ap.add_argument("--chambers", default="auto", choices=["auto", "dl", "rules"],
                    help="chamber labels: deep learning (segment_heart_dl.py) when available, or the rules")
    a = ap.parse_args()
    main(a.bpm, a.faces, chambers=a.chambers)
