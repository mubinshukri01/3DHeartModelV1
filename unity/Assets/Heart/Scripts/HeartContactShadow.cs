// UTM-Claude Research Project | rohayanti@utm.my
// A soft contact shadow under the heart, on the desk in the demo scene or on the real table in mixed reality,
// so the heart looks like it sits there. Built from the meshes' rest shape in the heart's current pose, on a
// grid seen from above: a dark core where the heart comes within a few centimetres of the desk, and a wide,
// faint footprint under the whole heart (the room light it blocks), fading as the heart is lifted. No
// real-time shadow maps, so it costs next to nothing in VR. HeartManipulator rebuilds it after a change.
using System.Collections.Generic;
using UnityEngine;

namespace UTM.Heart
{
    /// <summary>
    /// Put this on the root of an imported heart_animated.glb. In play mode it builds the shadow when enabled;
    /// tools call <see cref="Build"/>. The desk is HeartManipulator's (or the plane just under the heart's lowest
    /// point), and the shadow lies flat on it in the space of the heart's parent, whatever the heart's pose.
    /// </summary>
    [DisallowMultipleComponent]
    public class HeartContactShadow : MonoBehaviour
    {
        public const int Resolution = 128;            // shadow texture, cells across the square under the heart

        [Tooltip("Heart/Contact Shadow")]
        public Shader shader;
        [Range(0f, 1f)] public float opacity = 0.85f;
        [Tooltip("Height above the desk (m) over which the dark core fades out")]
        public float contactHeight = 0.03f;
        [Range(0f, 1f), Tooltip("Darkness of the wide footprint under the whole heart")]
        public float footprint = 0.35f;
        [Tooltip("Blur of the dark core (m)")]
        public float coreBlur = 0.004f;
        [Tooltip("Blur of the footprint (m)")]
        public float footprintBlur = 0.015f;
        [Tooltip("Gap between the heart's lowest point and the desk (m)")]
        public float gap = 0.0005f;
        [Tooltip("Lift above the desk (m) at which the footprint has faded out")]
        public float liftFade = 0.08f;

        GameObject m_Quad;
        Texture2D m_Texture;
        Material m_Material;

        public GameObject Quad => m_Quad;
        public Texture2D Texture => m_Texture;
        /// <summary>Height of the desk in the space of the heart's parent (m).</summary>
        public float DeskHeight { get; private set; }

        void OnEnable()
        {
            if (Application.isPlaying) Build();
        }

        void OnDisable() => Clear();

        /// <summary>All vertices of the heart's meshes (rest shape) in the root's local space.</summary>
        public static List<Vector3> LocalVertices(Transform root) => Vertices(root, root);

        /// <summary>All vertices of the heart's meshes (rest shape, current pose) in a transform's space (null: world).</summary>
        public static List<Vector3> Vertices(Transform root, Transform space)
        {
            var points = new List<Vector3>();
            var toSpace = space != null ? space.worldToLocalMatrix : Matrix4x4.identity;
            foreach (var smr in root.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                if (smr.sharedMesh == null) continue;
                var m = toSpace * smr.transform.localToWorldMatrix;
                foreach (var v in smr.sharedMesh.vertices) points.Add(m.MultiplyPoint3x4(v));
            }
            return points;
        }

        /// <summary>(Re)build the shadow texture and the quad that shows it.</summary>
        public void Build()
        {
            Clear();
            var stage = transform.parent;
            var points = Vertices(transform, stage);
            if (points.Count == 0) return;
            float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, minZ = float.MaxValue, maxZ = float.MinValue;
            foreach (var p in points)
            {
                minX = Mathf.Min(minX, p.x); maxX = Mathf.Max(maxX, p.x);
                minZ = Mathf.Min(minZ, p.z); maxZ = Mathf.Max(maxZ, p.z);
                minY = Mathf.Min(minY, p.y);
            }
            var manipulator = GetComponent<HeartManipulator>();
            DeskHeight = manipulator != null && manipulator.IsReady ? manipulator.DeskHeight : minY - gap;
            var lift = Mathf.Clamp01((minY - gap - DeskHeight) / liftFade);
            var footprintNow = footprint * (1f - lift * lift * (3f - 2f * lift));
            var size = Mathf.Max(maxX - minX, maxZ - minZ) + 6f * footprintBlur;     // room for the blur to fade out
            var x0 = (minX + maxX - size) * 0.5f;
            var z0 = (minZ + maxZ - size) * 0.5f;
            var cell = size / Resolution;

            // under each cell: how close the heart comes to the desk
            var n = Resolution;
            var lowest = new float[n * n];
            for (var k = 0; k < lowest.Length; k++) lowest[k] = float.PositiveInfinity;
            foreach (var p in points)
            {
                var i = Mathf.Clamp((int)((p.x - x0) / cell), 0, n - 1);
                var j = Mathf.Clamp((int)((p.z - z0) / cell), 0, n - 1);
                lowest[j * n + i] = Mathf.Min(lowest[j * n + i], p.y - DeskHeight);
            }
            var core = new float[n * n];
            var foot = new float[n * n];
            for (var k = 0; k < lowest.Length; k++)
            {
                if (float.IsInfinity(lowest[k])) continue;
                var t = Mathf.Clamp01(lowest[k] / contactHeight);
                core[k] = 1f - t * t * (3f - 2f * t);
                foot[k] = 1f;
            }
            Blur(core, n, coreBlur / cell);
            Blur(foot, n, footprintBlur / cell);
            var alpha = new byte[n * n];
            for (var k = 0; k < alpha.Length; k++)
                alpha[k] = (byte)Mathf.RoundToInt(255f * Mathf.Clamp01(1f - (1f - core[k]) * (1f - footprintNow * foot[k])));

            m_Texture = new Texture2D(n, n, TextureFormat.Alpha8, false)
                { name = "Heart contact shadow", wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear, hideFlags = HideFlags.DontSave };
            m_Texture.SetPixelData(alpha, 0);       // row 0 = v 0 = the -Z edge, x along u
            m_Texture.Apply(false);
            m_Material = new Material(shader) { name = "Heart contact shadow", hideFlags = HideFlags.DontSave };
            m_Material.mainTexture = m_Texture;
            m_Material.SetFloat("_Opacity", opacity);

            var mesh = new Mesh { name = "Heart contact shadow", hideFlags = HideFlags.DontSave };
            mesh.vertices = new[]
            {
                new Vector3(x0, DeskHeight, z0), new Vector3(x0 + size, DeskHeight, z0),
                new Vector3(x0, DeskHeight, z0 + size), new Vector3(x0 + size, DeskHeight, z0 + size),
            };
            mesh.uv = new[] { new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1), new Vector2(1, 1) };
            mesh.triangles = new[] { 0, 2, 3, 0, 3, 1 };
            mesh.normals = new[] { Vector3.up, Vector3.up, Vector3.up, Vector3.up };
            m_Quad = new GameObject("Contact shadow", typeof(MeshFilter), typeof(MeshRenderer)) { hideFlags = HideFlags.DontSave };
            m_Quad.transform.SetParent(stage, false);                // lies on the desk, not on the heart
            m_Quad.GetComponent<MeshFilter>().sharedMesh = mesh;
            var r = m_Quad.GetComponent<MeshRenderer>();
            r.sharedMaterial = m_Material;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;
        }

        public void Clear()
        {
            if (m_Quad != null) Discard(m_Quad.GetComponent<MeshFilter>().sharedMesh);
            Discard(m_Quad);
            Discard(m_Material);
            Discard(m_Texture);
            m_Quad = null; m_Material = null; m_Texture = null;
        }

        static void Discard(Object o)
        {
            if (o == null) return;
            if (Application.isPlaying) Destroy(o); else DestroyImmediate(o);
        }

        /// <summary>Approximate Gaussian blur (three box passes each way) of an n x n grid, radius in cells.</summary>
        static void Blur(float[] g, int n, float radius)
        {
            var r = Mathf.Max(1, Mathf.RoundToInt(radius));              // three boxes of half-width r ~ Gaussian, sigma r
            var tmp = new float[g.Length];
            for (var pass = 0; pass < 3; pass++)
            {
                BoxPass(g, tmp, n, r, 1, n);      // along x
                BoxPass(tmp, g, n, r, n, 1);      // along z
            }
        }

        static void BoxPass(float[] src, float[] dst, int n, int r, int step, int lineStep)
        {
            for (var line = 0; line < n; line++)
            {
                var start = line * lineStep;
                var sum = 0f;
                for (var i = -r; i <= r; i++) sum += i >= 0 && i < n ? src[start + i * step] : 0f;
                for (var i = 0; i < n; i++)
                {
                    dst[start + i * step] = sum / (2 * r + 1);
                    var add = i + r + 1; var drop = i - r;
                    if (add < n) sum += src[start + add * step];
                    if (drop >= 0) sum -= src[start + drop * step];
                }
            }
        }
    }
}
