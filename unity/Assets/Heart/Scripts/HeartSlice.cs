// UTM-Claude Research Project | rohayanti@utm.my
// The slice tool: a cutting plane through the heart, in the heart's own axial, coronal or sagittal direction
// (the CT's axes), so the cut stays anatomical however the heart is turned. Everything on one side is cut
// away (all tissue shaders read the plane), and the cut face shows the patient's CT from the case package,
// lined up with the anatomy because both come from the same grid, or the structures' colours.
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace UTM.Heart
{
    /// <summary>Put this on the root of an imported heart_animated.glb. The CT comes from the case package.</summary>
    [DisallowMultipleComponent]
    public class HeartSlice : MonoBehaviour
    {
        public enum Orientation { Axial, Coronal, Sagittal }

        static readonly int k_Plane = Shader.PropertyToID("_HeartClipPlane");

        [Tooltip("Cut the heart")]
        public bool active;
        public Orientation orientation = Orientation.Axial;
        [Range(-1f, 1f), Tooltip("Where the plane crosses the heart: -1 .. 1 of its size along the plane's normal")]
        public float position;
        [Tooltip("Keep the other side")]
        public bool flip;
        [Tooltip("Show the CT on the cut face (otherwise the structures' colours)")]
        public bool showCt = true;
        [Range(0f, 1f), Tooltip("Structure colour over the CT")]
        public float tint = 0.3f;
        public float windowCenter = 300f, windowWidth = 800f;
        [Tooltip("Heart/Slice Cap")]
        public Shader capShader;
        [Tooltip("Case package in Assets/StreamingAssets/Cases/")]
        public string caseId = "DEMO-001";

        HeartCase m_Case;
        ushort[] m_CtValues;
        byte[] m_LabelValues;
        Texture3D m_Ct, m_Labels, m_Mask;
        Texture2D m_Palette;
        Material m_CapMaterial;
        GameObject m_Cap;
        Bounds m_Bounds;                               // the heart in its own space (rest shape)
        bool m_Ready;

        public HeartCase Case => m_Case;
        public string CaseError { get; private set; }
        public GameObject Cap => m_Cap;
        public Material CapMaterial => m_CapMaterial;
        public Bounds LocalBounds => m_Bounds;

        void OnEnable()
        {
            if (Application.isPlaying) Init();
        }

        void OnDisable()
        {
            Shader.SetGlobalVector(k_Plane, Vector4.zero);
            if (m_Cap != null) m_Cap.SetActive(false);
        }

        void OnDestroy() => Clear();

        void LateUpdate() => Apply();

        /// <summary>Load the case package and build the cut face (called on enable in play mode).</summary>
        public void Init()
        {
            Clear();
            var points = HeartContactShadow.LocalVertices(transform);
            m_Bounds = new Bounds(points.Count > 0 ? points[0] : Vector3.zero, Vector3.zero);
            foreach (var p in points) m_Bounds.Encapsulate(p);
            try
            {
                m_Case = HeartCase.Load(HeartCase.StreamingFolder(caseId));
                m_CtValues = m_Case.ReadCt();
                m_LabelValues = m_Case.ReadLabels();
                m_Ct = m_Case.CtTexture(m_CtValues);
                m_Labels = m_Case.LabelTexture(m_LabelValues);
                m_Mask = Mask(m_Case, m_LabelValues);
                m_Palette = Palette(m_Case);
                CaseError = null;
            }
            catch (Exception e)
            {
                m_Case = null;
                CaseError = e.Message;
            }
            m_CapMaterial = new Material(capShader) { name = "Heart slice face", hideFlags = HideFlags.DontSave };
            if (m_Case != null)
            {
                m_CapMaterial.SetTexture("_CtTex", m_Ct);
                m_CapMaterial.SetTexture("_LabelTex", m_Labels);
                m_CapMaterial.SetTexture("_MaskTex", m_Mask);
                m_CapMaterial.SetTexture("_Palette", m_Palette);
                m_CapMaterial.SetVector("_Dims", new Vector4(m_Case.grid.dims[0], m_Case.grid.dims[1], m_Case.grid.dims[2], 1f));
                m_CapMaterial.SetFloat("_HuOffset", m_Case.ct.hu_offset);
            }
            var mesh = new Mesh { name = "Heart slice face", hideFlags = HideFlags.DontSave };
            mesh.vertices = new[] { new Vector3(-0.5f, -0.5f, 0f), new Vector3(0.5f, -0.5f, 0f), new Vector3(-0.5f, 0.5f, 0f), new Vector3(0.5f, 0.5f, 0f) };
            mesh.triangles = new[] { 0, 2, 3, 0, 3, 1 };
            mesh.RecalculateNormals();
            m_Cap = new GameObject("Slice face", typeof(MeshFilter), typeof(MeshRenderer)) { hideFlags = HideFlags.DontSave };
            m_Cap.transform.SetParent(transform, false);
            m_Cap.GetComponent<MeshFilter>().sharedMesh = mesh;
            var r = m_Cap.GetComponent<MeshRenderer>();
            r.sharedMaterial = m_CapMaterial;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            m_Cap.SetActive(false);
            m_Ready = true;
            Apply();
        }

        /// <summary>Unit normal of the plane in the heart's space, pointing into the part that is kept.</summary>
        public Vector3 LocalNormal
        {
            get
            {
                // kept by default: below an axial cut, behind a coronal cut, the patient's left of a sagittal cut,
                // so the cut face looks up, forwards (anterior) or to the patient's right
                var n = orientation == Orientation.Axial ? Vector3.down : orientation == Orientation.Coronal ? Vector3.back : Vector3.left;
                return flip ? -n : n;
            }
        }

        /// <summary>The point where the plane crosses the heart, in the heart's space.</summary>
        public Vector3 LocalPoint
        {
            get
            {
                var axis = orientation == Orientation.Axial ? Vector3.up : orientation == Orientation.Coronal ? Vector3.forward : Vector3.right;
                var extent = Vector3.Dot(m_Bounds.extents, new Vector3(Mathf.Abs(axis.x), Mathf.Abs(axis.y), Mathf.Abs(axis.z)));
                return m_Bounds.center + axis * (position * extent);
            }
        }

        /// <summary>The cut as a world plane (normal into the kept part).</summary>
        public Plane WorldPlane
        {
            get
            {
                var n = transform.TransformDirection(LocalNormal).normalized;
                return new Plane(n, transform.TransformPoint(LocalPoint));
            }
        }

        /// <summary>True when a world point is cut away.</summary>
        public bool IsCut(Vector3 world) => active && WorldPlane.GetDistanceToPoint(world) < 0f;

        public void Apply()
        {
            if (!m_Ready) return;
            if (!active)
            {
                Shader.SetGlobalVector(k_Plane, Vector4.zero);
                m_Cap.SetActive(false);
                return;
            }
            var plane = WorldPlane;
            Shader.SetGlobalVector(k_Plane, new Vector4(plane.normal.x, plane.normal.y, plane.normal.z, plane.distance));
            m_Cap.SetActive(m_Case != null);
            if (m_Case == null) return;
            // the face: a square on the plane, facing the cut-away side, a little larger than the heart
            var n = LocalNormal;
            var size = 2.4f * m_Bounds.extents.magnitude;
            m_Cap.transform.localPosition = LocalPoint;
            m_Cap.transform.localRotation = Quaternion.LookRotation(n, Mathf.Abs(n.y) > 0.9f ? Vector3.forward : Vector3.up);
            m_Cap.transform.localScale = new Vector3(size, size, 1f);
            m_CapMaterial.SetMatrix("_VoxelFromWorld", m_Case.VoxelFromLocal * transform.worldToLocalMatrix);
            m_CapMaterial.SetVector("_Window", new Vector4(windowCenter, windowWidth, 0f, 0f));
            m_CapMaterial.SetFloat("_ShowCt", showCt ? 1f : 0f);
            m_CapMaterial.SetFloat("_Tint", tint);
        }

        // ── the volumes on the CPU (labels for the pointer, the 2D CT view) ─────────────────
        /// <summary>Structure label at a world point (0 = none or no case package).</summary>
        public int LabelAt(Vector3 world)
        {
            if (m_Case == null) return 0;
            var k = m_Case.VoxelIndex(transform.InverseTransformPoint(world));
            return k >= 0 ? m_LabelValues[k] : 0;
        }

        /// <summary>
        /// The CT in the plane of the cut, as a greyscale image of the whole scan (current window), with the
        /// voxel the plane passes through. Axis-aligned, because the cut follows the CT's own axes.
        /// </summary>
        public Texture2D CtImage(Texture2D reuse = null)
        {
            if (m_Case == null) return reuse;
            var d = m_Case.grid.dims;
            var v = m_Case.LocalToVoxel(LocalPoint);
            // voxel axes: x = patient left (Unity -X), y = posterior (Unity -Z), z = superior (Unity +Y)
            int ax = orientation == Orientation.Axial ? 2 : orientation == Orientation.Coronal ? 1 : 0;
            int k = Mathf.Clamp(Mathf.RoundToInt(v[ax]), 0, d[ax] - 1);
            int u = ax == 0 ? 1 : 0, w = ax == 2 ? 1 : 2;                    // image columns, rows
            int width = d[u], height = d[w];
            if (reuse == null || reuse.width != width || reuse.height != height)
                reuse = new Texture2D(width, height, TextureFormat.RGBA32, false) { name = "CT view", wrapMode = TextureWrapMode.Clamp, hideFlags = HideFlags.DontSave };
            var px = new Color32[width * height];
            var lo = windowCenter - 0.5f * windowWidth;
            var idx = new int[3];
            for (var row = 0; row < height; row++)
            for (var col = 0; col < width; col++)
            {
                idx[ax] = k; idx[u] = col; idx[w] = row;
                // radiological view: the patient's right (voxel x = 0) on the image's left; axial: anterior up, so y
                // (posterior) runs downwards; coronal and sagittal: superior up; sagittal: anterior on the left
                if (ax == 2) idx[1] = d[1] - 1 - row;
                var hu = m_CtValues[idx[0] + d[0] * (idx[1] + d[1] * idx[2])] + m_Case.ct.hu_offset;
                var g = (byte)Mathf.RoundToInt(255f * Mathf.Clamp01((hu - lo) / windowWidth));
                px[row * width + col] = new Color32(g, g, g, 255);
            }
            reuse.SetPixels32(px);
            reuse.Apply(false);
            return reuse;
        }

        static Texture3D Mask(HeartCase c, byte[] labels)
        {
            var inside = new byte[labels.Length];
            for (var i = 0; i < labels.Length; i++) inside[i] = labels[i] > 0 ? (byte)255 : (byte)0;
            var t = new Texture3D(c.grid.dims[0], c.grid.dims[1], c.grid.dims[2], GraphicsFormat.R8_UNorm, TextureCreationFlags.None)
                { name = "Heart inside", wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear, hideFlags = HideFlags.DontSave };
            t.SetPixelData(inside, 0);
            t.Apply(false);
            return t;
        }

        static Texture2D Palette(HeartCase c)
        {
            var px = new Color32[256];
            foreach (var s in c.structures)
                if (s.label > 0 && s.label < 256) px[s.label] = s.Color;
            var t = new Texture2D(256, 1, TextureFormat.RGBA32, false)
                { name = "Label colours", filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp, hideFlags = HideFlags.DontSave };
            t.SetPixels32(px);
            t.Apply(false);
            return t;
        }

        void Clear()
        {
            foreach (var o in new UnityEngine.Object[] { m_Ct, m_Labels, m_Mask, m_Palette, m_CapMaterial,
                                                         m_Cap != null ? m_Cap.GetComponent<MeshFilter>().sharedMesh : null, m_Cap })
                if (o != null) { if (Application.isPlaying) Destroy(o); else DestroyImmediate(o); }
            m_Ready = false;
        }
    }
}
