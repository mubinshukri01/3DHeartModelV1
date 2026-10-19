// UTM-Claude Research Project | rohayanti@utm.my
// Realistic or labelled look for the imported heart. Realistic: procedural tissue shaders (Assets/Heart/
// Shaders; no texture coordinates needed) assigned by structure. Labelled: the GLB's own flat colours, on the
// same shaders with the detail turned off, so the slice tool can cut both looks.
using System;
using System.Collections.Generic;
using UnityEngine;

namespace UTM.Heart
{
    /// <summary>
    /// Put this on the root of an imported heart_animated.glb, next to HeartController. Materials come from
    /// Tools > Heart > Create demo scene (Assets/Heart/Materials). In play mode it also stores each vertex's
    /// rest position in the meshes' 4th UV channel, so the tissue pattern stays on the tissue as it beats.
    /// </summary>
    [DisallowMultipleComponent]
    public class HeartMaterials : MonoBehaviour
    {
        public enum Look { Realistic, Labelled }

        public const int RestUvChannel = 3;            // TEXCOORD3 in Assets/Heart/Shaders/HeartTissue.cginc

        [SerializeField] Look m_Look = Look.Realistic;

        [Header("Realistic look")]
        [Tooltip("Right ventricle (no separate wall label: its outer surface is the heart surface)")]
        public Material ventricleMuscle;
        [Tooltip("Left and right atria (outer surfaces)")]
        public Material atrialMuscle;
        [Tooltip("The LV wall ('Myocardium'); see-through, its opacity set by HeartController")]
        public Material wall;
        [Tooltip("LV blood pool, inside the wall")]
        public Material blood;
        [Tooltip("Aorta, descending aorta, pulmonary veins, coronary arteries")]
        public Material oxygenated;
        [Tooltip("Venae cavae, pulmonary artery")]
        public Material deoxygenated;
        public Material fat;

        [Serializable]
        struct Original { public Renderer renderer; public Material[] materials; }
        [SerializeField, HideInInspector] List<Original> m_Labelled = new List<Original>();
        readonly List<Mesh> m_RestMeshes = new List<Mesh>();
        readonly Dictionary<Renderer, Material> m_Flat = new Dictionary<Renderer, Material>();
        static readonly int k_BaseColor = Shader.PropertyToID("baseColorFactor");

        /// <summary>Structure name -> its material in the realistic look (null: keep the GLB's).</summary>
        public Material MaterialFor(string structure)
        {
            switch (structure)
            {
                case "Myocardium": return wall;
                case "LV blood pool": return blood;
                case "Right ventricle": return ventricleMuscle;
                case "LA blood pool": case "Right atrium": return atrialMuscle;
                case "Aorta": case "Descending aorta": case "Pulmonary veins": case "Coronary arteries": return oxygenated;
                case "Pulmonary artery": case "Superior vena cava": case "Inferior vena cava": return deoxygenated;
                case "Epicardial fat (EAT)": return fat;
                default: return null;
            }
        }

        public Look CurrentLook
        {
            get => m_Look;
            set { m_Look = value; Apply(); }
        }

        void OnEnable()
        {
            if (!Application.isPlaying) return;      // in the Editor, Apply is called by the tools (materials stay assets)
            StoreRestPositions();
            Apply();
        }

        void OnValidate()
        {
            if (Application.isPlaying && isActiveAndEnabled) Apply();    // the Look field in the Inspector, during Play
        }

        void OnDestroy()
        {
            foreach (var m in m_RestMeshes)
                if (Application.isPlaying) Destroy(m); else DestroyImmediate(m);
            foreach (var m in m_Flat.Values)
                if (Application.isPlaying) Destroy(m); else DestroyImmediate(m);
        }

        /// <summary>Assign the materials of the current look to every structure.</summary>
        public void Apply()
        {
            foreach (var r in GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                var i = m_Labelled.FindIndex(o => o.renderer == r);
                if (i < 0)
                {
                    m_Labelled.Add(new Original { renderer = r, materials = r.sharedMaterials });
                    i = m_Labelled.Count - 1;
                }
                var realistic = MaterialFor(r.name);
                if (realistic == null) r.sharedMaterials = m_Labelled[i].materials;
                else r.sharedMaterials = new[] { m_Look == Look.Realistic ? realistic : Flat(r, realistic, m_Labelled[i].materials) };
            }
        }

        /// <summary>The labelled look of a structure: its tissue shader with the GLB's colour and no detail.</summary>
        Material Flat(Renderer r, Material realistic, Material[] gltf)
        {
            if (m_Flat.TryGetValue(r, out var m) && m != null) return m;
            m = new Material(realistic) { name = r.name + " (labelled)", hideFlags = HideFlags.DontSave };
            var src = gltf != null && gltf.Length > 0 ? gltf[0] : null;
            if (src != null && src.HasProperty(k_BaseColor)) m.SetColor(k_BaseColor, src.GetColor(k_BaseColor));
            m.SetFloat("_Detail", 0f);
            m.SetFloat("_BumpMm", 0f);
            m.SetFloat("_Gloss", 0.45f);
            m.SetFloat("_Studio", 0.4f);
            m_Flat[r] = m;
            return m;
        }

        /// <summary>
        /// Copy each mesh with its rest positions in UV channel <see cref="RestUvChannel"/> (skinned meshes' blend
        /// shapes move the positions but not the UVs). Runtime copies; the imported meshes are not changed.
        /// </summary>
        public void StoreRestPositions()
        {
            foreach (var smr in GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                var src = smr.sharedMesh;
                if (src == null || m_RestMeshes.Contains(src)) continue;
                var copy = Instantiate(src);
                copy.name = src.name + " (rest positions)";
                copy.SetUVs(RestUvChannel, src.vertices);
                smr.sharedMesh = copy;
                m_RestMeshes.Add(copy);
            }
        }
    }
}
