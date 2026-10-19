// UTM-Claude Research Project | rohayanti@utm.my
// Which structure is under a ray (the pointer now, a hand's ray later). Mesh colliders on the rest shape
// follow the heart's pose; hidden structures and the part cut away by the slice tool are skipped, and on the
// slice's cut face the structure comes from the case package's labels.
using System.Collections.Generic;
using UnityEngine;

namespace UTM.Heart
{
    [DisallowMultipleComponent]
    public class HeartPicker : MonoBehaviour
    {
        public struct Hit
        {
            public string structure;
            public Vector3 point;          // world
            public bool onSliceFace;
        }

        readonly Dictionary<Collider, SkinnedMeshRenderer> m_Owner = new Dictionary<Collider, SkinnedMeshRenderer>();
        readonly List<GameObject> m_Objects = new List<GameObject>();
        HeartSlice m_Slice;

        void OnEnable()
        {
            if (Application.isPlaying) Build();
        }

        void OnDestroy() => Clear();

        /// <summary>Add a mesh collider (rest shape) for every structure (called on enable in play mode).</summary>
        public void Build()
        {
            Clear();
            m_Slice = GetComponent<HeartSlice>();
            foreach (var smr in GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                if (smr.sharedMesh == null) continue;
                var go = new GameObject("Pick " + smr.name) { hideFlags = HideFlags.DontSave };
                go.transform.SetParent(smr.transform, false);
                var c = go.AddComponent<MeshCollider>();
                c.sharedMesh = smr.sharedMesh;
                m_Owner[c] = smr;
                m_Objects.Add(go);
            }
        }

        public bool Pick(Ray ray, out Hit hit)
        {
            hit = default;
            Physics.SyncTransforms();                                    // the heart may have moved this frame
            var best = float.PositiveInfinity;
            foreach (var h in Physics.RaycastAll(ray, 100f))
            {
                if (!m_Owner.TryGetValue(h.collider, out var smr) || h.distance >= best) continue;
                if (!smr.enabled || !smr.gameObject.activeInHierarchy) continue;
                if (m_Slice != null && m_Slice.IsCut(h.point)) continue;
                best = h.distance;
                hit = new Hit { structure = smr.name, point = h.point };
            }
            // the slice's cut face, when the ray meets it first
            if (m_Slice != null && m_Slice.active && m_Slice.Case != null && m_Slice.WorldPlane.Raycast(ray, out var t) && t < best)
            {
                var p = ray.GetPoint(t);
                var s = m_Slice.Case.StructureByLabel(m_Slice.LabelAt(p));
                if (s != null && Visible(s.name))
                {
                    hit = new Hit { structure = s.name, point = p, onSliceFace = true };
                    best = t;
                }
            }
            return !float.IsPositiveInfinity(best);
        }

        bool Visible(string structure)
        {
            foreach (var smr in m_Owner.Values)
                if (smr.name == structure) return smr.enabled && smr.gameObject.activeInHierarchy;
            return true;
        }

        void Clear()
        {
            foreach (var go in m_Objects)
                if (go != null) { if (Application.isPlaying) Destroy(go); else DestroyImmediate(go); }
            m_Objects.Clear();
            m_Owner.Clear();
        }
    }
}
