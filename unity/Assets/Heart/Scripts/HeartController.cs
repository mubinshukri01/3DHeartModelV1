// UTM-Claude Research Project | rohayanti@utm.my
// Unity side of the 3D heart model (heart_animated.glb imported with glTFast): plays the CardiacCycle
// morph-target animation and offers the web viewer's controls (heart rate, pause, structure
// visibility, wall opacity, current phase) for desktop and XR scenes.
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

namespace UTM.Heart
{
    /// <summary>
    /// Put this on the root of an imported heart_animated.glb, the object with the Animator that
    /// glTFast adds. The clip is played through a PlayableGraph, so no Animator Controller asset is
    /// needed (glTFast's Editor import does not create one). A heart loaded at runtime (HeartStage) has
    /// glTFast's legacy Animation instead, the only kind that works in builds; that is played directly.
    /// </summary>
    [RequireComponent(typeof(Animator))]
    public class HeartController : MonoBehaviour
    {
        static readonly int k_BaseColor = Shader.PropertyToID("baseColorFactor");   // glTFast shaders

        // Same phase names and boundaries (fraction of the R-R interval) as the web viewer
        static readonly (float end, string name)[] k_Phases =
        {
            (0.05f, "Isovolumic contraction"), (0.33f, "Ejection (systole)"), (0.42f, "Isovolumic relaxation"),
            (0.60f, "Rapid filling"), (0.80f, "Diastasis (CT rest pose)"), (0.98f, "Atrial systole"),
            (1.01f, "Isovolumic contraction"),
        };

        [Tooltip("CardiacCycle clip imported from heart_animated.glb")]
        public AnimationClip cardiacCycle;

        [Tooltip("Heart rate the clip was built for (72 unless build_heart_model.py --bpm was changed)")]
        public float clipBpm = 72f;

        [Range(30f, 150f)] public float heartRate = 72f;
        public bool playing = true;

        [Range(0f, 1f)] public float wallOpacity = 0.45f;

        [Tooltip("Structure whose opacity follows Wall Opacity; its material must be transparent (BLEND) in the GLB")]
        public string wallStructure = "Myocardium";

        [Tooltip("Small on-screen panel for desktop testing (not visible inside a headset)")]
        public bool showDebugPanel = true;

        [Tooltip("Height of the on-screen panel as a share of the Game view; it scales with the view")]
        [Range(0.3f, 1f)] public float panelHeight = 0.75f;

        const float k_PanelWidth = 280f, k_PanelLayoutHeight = 520f;   // panel layout size before scaling

        PlayableGraph m_Graph;
        AnimationClipPlayable m_Clip;
        Animation m_Legacy;
        AnimationState m_State;
        MaterialPropertyBlock m_Block;
        readonly Dictionary<string, Renderer> m_Structures = new Dictionary<string, Renderer>();

        /// <summary>Position in the beat as a fraction of the R-R interval (0 = R wave).</summary>
        public float Phase => m_State != null && m_State.length > 0f ? Mathf.Repeat(m_State.time, m_State.length) / m_State.length
            : m_Clip.IsValid() && cardiacCycle != null && cardiacCycle.length > 0f
                ? (float)(m_Clip.GetTime() % cardiacCycle.length / cardiacCycle.length)
                : 0f;

        /// <summary>Name of the cardiac phase at <see cref="Phase"/>.</summary>
        public string PhaseName
        {
            get
            {
                var p = Phase;
                foreach (var (end, name) in k_Phases)
                    if (p < end) return name;
                return k_Phases[0].name;
            }
        }

        /// <summary>Playback speed of the clip (heart rate / clip heart rate, 0 when paused).</summary>
        public double Speed => m_State != null ? m_State.speed : m_Clip.IsValid() ? m_Clip.GetSpeed() : 0.0;

        /// <summary>Renderer of each structure, by name (e.g. "LV blood pool", "Right atrium").</summary>
        public IReadOnlyDictionary<string, Renderer> Structures
        {
            get
            {
                if (m_Structures.Count == 0)
                    foreach (var r in GetComponentsInChildren<SkinnedMeshRenderer>(true))   // the model's meshes only
                        m_Structures[r.name] = r;
                return m_Structures;
            }
        }

        void OnEnable() => Play();
        void OnDisable() => Stop();
        void Update() => Apply();

        /// <summary>Start the cardiac cycle (called automatically when the component is enabled).</summary>
        public void Play()
        {
            if (m_Graph.IsValid() || m_State != null) return;
            var legacy = GetComponent<Animation>();
            if (cardiacCycle == null && legacy != null && legacy.clip != null)
            {
                m_Legacy = legacy;
                m_State = legacy[legacy.clip.name];
                m_State.wrapMode = WrapMode.Loop;
                legacy.Play(legacy.clip.name);
                Apply();
                return;
            }
            if (cardiacCycle == null)
            {
                Debug.LogWarning("HeartController: assign the CardiacCycle clip from heart_animated.glb", this);
                return;
            }
            m_Clip = AnimationPlayableUtilities.PlayClip(GetComponent<Animator>(), cardiacCycle, out m_Graph);
            Apply();
        }

        public void Stop()
        {
            if (m_Graph.IsValid()) m_Graph.Destroy();
            if (m_State != null && m_Legacy != null) m_Legacy.Stop();
            m_State = null;
        }

        /// <summary>Push heart rate, pause state and wall opacity to the animation and the wall material.</summary>
        public void Apply()
        {
            if (m_Clip.IsValid()) m_Clip.SetSpeed(playing ? heartRate / clipBpm : 0.0);
            if (m_State != null) m_State.speed = playing ? heartRate / clipBpm : 0f;
            SetOpacity(wallStructure, wallOpacity);
        }

        /// <summary>Jump to a phase of the beat (0..1 of the R-R interval), e.g. 0.33 for end-systole.</summary>
        public void SetPhase(float phase)
        {
            if (m_State != null)
            {
                m_State.time = Mathf.Repeat(phase, 1f) * m_State.length;
                m_Legacy.Sample();
                return;
            }
            if (!m_Clip.IsValid()) return;
            m_Clip.SetTime(Mathf.Repeat(phase, 1f) * cardiacCycle.length);
            m_Graph.Evaluate();
        }

        /// <summary>Advance the animation by hand (edit mode, tests, frame-by-frame recording).</summary>
        public void Advance(float seconds)
        {
            if (m_Graph.IsValid()) m_Graph.Evaluate(seconds);
            if (m_State != null) { m_State.time += seconds * m_State.speed; m_Legacy.Sample(); }
        }

        public void SetVisible(string structure, bool visible)
        {
            if (Structures.TryGetValue(structure, out var r)) r.enabled = visible;
        }

        /// <summary>
        /// Opacity of a structure with a transparent material. Uses a MaterialPropertyBlock, so the
        /// imported material is not copied or changed.
        /// </summary>
        public void SetOpacity(string structure, float alpha)
        {
            if (!Structures.TryGetValue(structure, out var r) || r.sharedMaterial == null) return;
            m_Block ??= new MaterialPropertyBlock();
            r.GetPropertyBlock(m_Block);
            var mat = r.sharedMaterial;
            var c = mat.HasProperty(k_BaseColor) ? mat.GetColor(k_BaseColor) : mat.color;
            c.a = alpha;
            m_Block.SetColor(k_BaseColor, c);
            r.SetPropertyBlock(m_Block);
        }

        float PanelScale => Mathf.Max(0.5f, panelHeight * Screen.height / k_PanelLayoutHeight);

        /// <summary>True when a screen point (pixels, origin bottom-left, like Input.mousePosition) is on the panel.</summary>
        public bool PanelContains(Vector2 screenPoint)
        {
            if (!showDebugPanel) return false;
            var s = PanelScale;
            return new Rect(10f * s, 10f * s, k_PanelWidth * s, k_PanelLayoutHeight * s)
                .Contains(new Vector2(screenPoint.x, Screen.height - screenPoint.y));
        }

        void OnGUI()
        {
            if (!showDebugPanel) return;
            var scale = PanelScale;
            var matrix = GUI.matrix;
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));     // also scales the mouse input
            GUILayout.BeginArea(new Rect(10, 10, k_PanelWidth, k_PanelLayoutHeight), GUI.skin.box);
            GUILayout.Label($"Heart rate {heartRate:0} bpm");
            heartRate = GUILayout.HorizontalSlider(heartRate, 30f, 150f);
            GUILayout.Label($"Wall opacity {wallOpacity:0.00}");
            wallOpacity = GUILayout.HorizontalSlider(wallOpacity, 0f, 1f);
            playing = GUILayout.Toggle(playing, "Playing");
            var look = GetComponent<HeartMaterials>();
            if (look != null)
            {
                var realistic = look.CurrentLook == HeartMaterials.Look.Realistic;
                if (GUILayout.Toggle(realistic, "Realistic look") != realistic)
                    look.CurrentLook = realistic ? HeartMaterials.Look.Labelled : HeartMaterials.Look.Realistic;
            }
            foreach (var kv in Structures)
            {
                var on = GUILayout.Toggle(kv.Value.enabled, kv.Key);
                if (on != kv.Value.enabled) kv.Value.enabled = on;
            }
            GUILayout.Label($"Phase {Phase * 100f:0}% R-R: {PhaseName}");
            GUILayout.Label("Rest pose = CT at 75% R-R; motion is procedural");
            if (GetComponent<HeartMouseInput>() != null) GUILayout.Label(HeartMouseInput.Hint);
            GUILayout.EndArea();
            GUI.matrix = matrix;
        }
    }
}
