// UTM-Claude Research Project | rohayanti@utm.my
// The CardioTwin XR interface (uGUI, built at runtime, so nothing but this script is needed):
//   * title bar: CardioTwin XR, the case picker (case packages in StreamingAssets/Cases), seated / standing,
//     desktop / headset quality (HeartStage), and the notice "Simulated motion · Research prototype"
//   * bottom toolbar: Rotate, Zoom, Slice (what a drag does), play / pause and heart rate, Reset view, Recentre
//   * Layers panel: heart, epicardial fat, CT on the cut, realistic look, wall opacity, each structure
//   * Case panel: the case package's card (fat volume, calcium score, "CAD model: in development", notice)
//   * slice controls with a 2D view of the CT in the plane of the cut
//   * a label with a callout line for the structure under the pointer
// Desktop: screen-space panels of frosted glass (Heart/UI Glass). Headset: the panels float in front of the
// viewer (renderMode = WorldSpace), plain translucent, and drift back in front when the viewer looks away.
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace UTM.Heart
{
    /// <summary>Put this on the root of an imported heart_animated.glb, with the other Heart components.</summary>
    [DisallowMultipleComponent]
    public class HeartUI : MonoBehaviour
    {
        public RenderMode renderMode = RenderMode.ScreenSpaceOverlay;
        [Tooltip("Camera for screen-space-camera or world-space panels (default: the main camera)")]
        public Camera uiCamera;
        [Tooltip("Label the structure under the pointer")]
        public bool hoverLabels = true;
        [Tooltip("Frosted-glass panels (desktop; not in world space)")]
        public bool glass = true;
        [Tooltip("Heart/UI Glass")]
        public Shader glassShader;
        [Tooltip("Headset: the panels drift back in front of the viewer when they look this far away (degrees)")]
        public float followAngle = 30f;

        public const string Notice = "Simulated motion · Research prototype";
        const float k_WorldDistance = 0.85f, k_WorldWidth = 1.0f, k_WorldDrop = 0.05f;   // headset panels (m)

        const string k_Fat = "Epicardial fat (EAT)";
        const float k_CtView = 260f;                   // width of the 2D CT view (reference pixels at 1080 high)
        static readonly Color k_Panel = new Color(0.07f, 0.09f, 0.12f, 0.84f);
        static readonly Color k_Accent = new Color(0.25f, 0.78f, 0.92f, 1f);
        static readonly Color k_Text = new Color(0.93f, 0.95f, 0.97f, 1f);
        static readonly Color k_Muted = new Color(0.62f, 0.68f, 0.75f, 1f);

        HeartController m_Controller;
        HeartManipulator m_Manipulator;
        HeartMouseInput m_Mouse;
        HeartSlice m_Slice;
        HeartMaterials m_Look;
        HeartPicker m_Picker;
        Canvas m_Canvas;
        Font m_Font;
        Sprite m_Round, m_Circle;
        Texture2D m_RoundTex, m_CircleTex, m_CtImage;
        Material m_Material, m_GlassMaterial;
        HeartStage m_Stage;
        Image m_PlayIcon;
        Text m_CaseButton;
        RectTransform m_CaseList;
        bool m_Following;
        readonly List<Image> m_Panels = new List<Image>();
        readonly Dictionary<string, Button> m_Buttons = new Dictionary<string, Button>();
        readonly Dictionary<string, Toggle> m_Toggles = new Dictionary<string, Toggle>();
        Slider m_Opacity, m_SlicePosition;
        Text m_Play, m_Rate, m_SliceText, m_CaseTitle, m_CalloutText;
        RawImage m_CtView;
        LayoutElement m_CtViewSize;
        RectTransform m_Callout, m_CalloutLine;
        string m_CtKey;

        public Canvas Canvas => m_Canvas;
        public RectTransform Toolbar { get; private set; }
        public RectTransform LayersPanel { get; private set; }
        public RectTransform CasePanel { get; private set; }
        public RectTransform SlicePanel { get; private set; }
        public RectTransform TitleBar { get; private set; }
        public RectTransform NoticePill { get; private set; }
        public RectTransform CaseList => m_CaseList;
        /// <summary>True when the panels use the frosted-glass material.</summary>
        public bool GlassOn => m_GlassMaterial != null && m_Panels.Count > 0 && m_Panels[0].material == m_GlassMaterial;
        public string TitleText => TitleBar == null ? "" : string.Join(" ", TitleBar.GetComponentsInChildren<Text>(true).Select(t => t.text));
        public IReadOnlyDictionary<string, Button> Buttons => m_Buttons;
        public IReadOnlyDictionary<string, Toggle> Toggles => m_Toggles;
        public Slider OpacitySlider => m_Opacity;
        public Slider SlicePositionSlider => m_SlicePosition;
        public RawImage CtView => m_CtView;
        public bool CalloutVisible => m_Callout != null && m_Callout.gameObject.activeSelf;
        public string CalloutText => m_CalloutText != null ? m_CalloutText.text : "";
        /// <summary>All the text of the Case panel (for checks and screen readers).</summary>
        public string CaseText => CasePanel == null ? "" : string.Join("\n", CasePanel.GetComponentsInChildren<Text>(true).Select(t => t.text));

        void Start()
        {
            if (Application.isPlaying) Build();       // after the other components have set up in OnEnable
        }

        void OnDestroy() => Clear();

        // ── build ─────────────────────────────────────────────────────────────
        public void Build()
        {
            Clear();
            m_Controller = GetComponent<HeartController>();
            m_Manipulator = GetComponent<HeartManipulator>();
            m_Mouse = GetComponent<HeartMouseInput>();
            m_Slice = GetComponent<HeartSlice>();
            m_Look = GetComponent<HeartMaterials>();
            m_Picker = GetComponent<HeartPicker>();
            m_Stage = GetComponentInParent<HeartStage>();
            if (m_Controller != null) m_Controller.showDebugPanel = false;     // this UI replaces the debug panel
            if (Application.isPlaying && EventSystem.current == null)
                new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));

            m_Font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            m_RoundTex = RoundedTexture(64, 14f);
            m_Round = Sprite.Create(m_RoundTex, new Rect(0, 0, 64, 64), new Vector2(0.5f, 0.5f), 100f, 0, SpriteMeshType.FullRect, new Vector4(16, 16, 16, 16));
            m_CircleTex = RoundedTexture(32, 16f);
            m_Circle = Sprite.Create(m_CircleTex, new Rect(0, 0, 32, 32), new Vector2(0.5f, 0.5f));

            var go = new GameObject("Heart UI", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster))
                { hideFlags = HideFlags.DontSave };
            m_Canvas = go.GetComponent<Canvas>();
            var scaler = go.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920f, 1080f);
            scaler.matchWidthOrHeight = 1f;
            SetRenderMode(renderMode);

            BuildTitle();
            BuildToolbar();
            BuildLayers();
            BuildCase();
            BuildSlicePanel();
            BuildCallout();
            // drawn after all the tissue (the see-through wall and fat come late in the transparent queue), so
            // panels in front of the heart are never painted over; still depth-tested in world space
            m_Material = new Material(Canvas.GetDefaultCanvasMaterial()) { name = "Heart UI", renderQueue = 4000, hideFlags = HideFlags.DontSave };
            foreach (var g in m_Canvas.GetComponentsInChildren<Graphic>(true)) g.material = m_Material;
            var shader = glassShader != null ? glassShader : Shader.Find("Heart/UI Glass");
            if (shader != null) m_GlassMaterial = new Material(shader) { name = "Heart UI glass", renderQueue = 4000, hideFlags = HideFlags.DontSave };
            ApplyGlass();
            Refresh();
            Canvas.ForceUpdateCanvases();
        }

        /// <summary>Frosted glass on the panels on the desktop; plain translucent in world space or without the shader.</summary>
        void ApplyGlass()
        {
            var on = glass && m_GlassMaterial != null && renderMode != RenderMode.WorldSpace;
            foreach (var p in m_Panels)
            {
                p.material = on ? m_GlassMaterial : m_Material;
                var c = p.color;
                c.a = on ? 0.55f : k_Panel.a;               // glass: how much tint over the blurred scene
                p.color = c;
            }
        }

        public void SetRenderMode(RenderMode mode)
        {
            renderMode = mode;
            if (m_Canvas == null) return;
            ApplyGlass();
            var cam = uiCamera != null ? uiCamera : Camera.main;
            m_Canvas.renderMode = mode;
            m_Canvas.worldCamera = cam;
            var rt = (RectTransform)m_Canvas.transform;
            if (mode == RenderMode.ScreenSpaceCamera) m_Canvas.planeDistance = cam != null ? Mathf.Max(cam.nearClipPlane * 2f, 0.05f) : 1f;
            if (mode == RenderMode.WorldSpace && cam != null)
            {
                // a 1 m wide sheet of panels 0.85 m in front of the viewer, upright, a little below the eyes
                rt.sizeDelta = new Vector2(1920f, 1080f);
                rt.localScale = Vector3.one * (k_WorldWidth / 1920f);
                var (position, rotation) = WorldPose(cam);
                rt.SetPositionAndRotation(position, rotation);
            }
        }

        static (Vector3, Quaternion) WorldPose(Camera cam)
        {
            var ahead = Vector3.ProjectOnPlane(cam.transform.forward, Vector3.up);
            ahead = ahead.sqrMagnitude > 1e-6f ? ahead.normalized : Vector3.forward;
            return (cam.transform.position + ahead * k_WorldDistance + Vector3.down * k_WorldDrop, Quaternion.LookRotation(ahead, Vector3.up));
        }

        /// <summary>
        /// Headset: when the viewer has turned more than <see cref="followAngle"/> away from the panels, they glide
        /// back in front (and stop once there), so they never chase small head movements.
        /// </summary>
        public void FollowStep(float dt)
        {
            if (m_Canvas == null || renderMode != RenderMode.WorldSpace) return;
            var cam = uiCamera != null ? uiCamera : Camera.main;
            if (cam == null) return;
            var rt = (RectTransform)m_Canvas.transform;
            var (position, rotation) = WorldPose(cam);
            var off = Vector3.Angle(Vector3.ProjectOnPlane(rt.position - cam.transform.position, Vector3.up), rotation * Vector3.forward);
            if (off > followAngle) m_Following = true;
            if (!m_Following) return;
            var t = 1f - Mathf.Exp(-4f * dt);
            rt.SetPositionAndRotation(Vector3.Lerp(rt.position, position, t), Quaternion.Slerp(rt.rotation, rotation, t));
            if (Vector3.Distance(rt.position, position) < 0.01f && Quaternion.Angle(rt.rotation, rotation) < 1f) m_Following = false;
        }

        void LateUpdate()
        {
            if (Application.isPlaying) FollowStep(Time.deltaTime);
        }

        void BuildTitle()
        {
            TitleBar = Panel("Title", m_Canvas.transform, new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -24f), 0f, false);
            Icon(TitleBar, "heart", 34f).color = new Color(0.93f, 0.33f, 0.38f, 1f);
            Label(TitleBar, "CardioTwin XR", 30, k_Text, TextAnchor.MiddleLeft, FontStyle.Bold);
            if (m_Stage != null)
            {
                Spacer(TitleBar, 10f);
                var caseButton = MakeButton(TitleBar, "Case", m_Slice != null && m_Slice.Case != null ? $"Case {m_Slice.Case.case_id}" : "Case", 250f,
                    () => m_CaseList.gameObject.SetActive(!m_CaseList.gameObject.activeSelf), 46f, "caret");
                m_CaseButton = caseButton.GetComponentInChildren<Text>();
                Spacer(TitleBar, 10f);
                MakeButton(TitleBar, "Seated", "Seated", 104f, () => m_Stage.SetPosture(HeartStage.Posture.Seated), 46f);
                MakeButton(TitleBar, "Standing", "Standing", 116f, () => m_Stage.SetPosture(HeartStage.Posture.Standing), 46f);
                Spacer(TitleBar, 10f);
                MakeButton(TitleBar, "Desktop", "Desktop", 112f, () => _ = m_Stage.SetQuality(HeartStage.Quality.Desktop), 46f);
                MakeButton(TitleBar, "Headset", "Headset", 112f, () => _ = m_Stage.SetQuality(HeartStage.Quality.Headset), 46f);
            }
            NoticePill = Panel("Notice", m_Canvas.transform, new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -112f), 0f, false);
            NoticePill.GetComponent<HorizontalLayoutGroup>().padding = new RectOffset(16, 16, 6, 6);
            Label(NoticePill, Notice, 19, k_Accent, TextAnchor.MiddleCenter);
            // the case picker: a list under the case button
            m_CaseList = Panel("Cases", m_Canvas.transform, new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Vector2(-150f, -112f), 300f, true);
            Label(m_CaseList, "Case packages", 19, k_Muted, TextAnchor.MiddleLeft);
            foreach (var id in HeartStage.AvailableCases())
            {
                var caseId = id;
                MakeButton(m_CaseList, "Case " + caseId, caseId, 0f, () =>
                {
                    m_CaseList.gameObject.SetActive(false);
                    if (m_Stage != null && caseId != m_Stage.CaseId) _ = m_Stage.LoadCase(caseId);
                }, 44f);
            }
            m_CaseList.gameObject.SetActive(false);
        }

        void BuildToolbar()
        {
            Toolbar = Panel("Toolbar", m_Canvas.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(0f, 24f), 0f, false);
            MakeButton(Toolbar, "Rotate", "Rotate", 132f, () => SetMode(HeartMouseInput.DragMode.Rotate), 52f, "rotate");
            MakeButton(Toolbar, "Zoom", "Zoom", 120f, () => SetMode(HeartMouseInput.DragMode.Zoom), 52f, "zoom");
            MakeButton(Toolbar, "Slice", "Slice", 118f, ToggleSlice, 52f, "slice");
            Spacer(Toolbar, 10f);
            var play = MakeButton(Toolbar, "Play", "Pause", 132f, () => { if (m_Controller != null) m_Controller.playing = !m_Controller.playing; }, 52f, "pause");
            m_Play = play.GetComponentInChildren<Text>();
            m_PlayIcon = play.transform.Find("Icon").GetComponent<Image>();
            MakeButton(Toolbar, "Slower", "−", 52f, () => ChangeRate(-5f));
            m_Rate = Label(Toolbar, "72 bpm", 24, k_Text, TextAnchor.MiddleCenter);
            m_Rate.gameObject.AddComponent<LayoutElement>().preferredWidth = 100f;
            MakeButton(Toolbar, "Faster", "+", 52f, () => ChangeRate(5f));
            Spacer(Toolbar, 10f);
            MakeButton(Toolbar, "Reset", "Reset", 124f, ResetView, 52f, "reset");
            if (m_Stage != null) MakeButton(Toolbar, "Recentre", "Recentre", 150f, () => m_Stage.Recentre(), 52f, "recentre");
        }

        void BuildLayers()
        {
            LayersPanel = Panel("Layers", m_Canvas.transform, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(24f, -24f), 340f, true);
            Label(LayersPanel, "Layers", 30, k_Text, TextAnchor.MiddleLeft, FontStyle.Bold);
            MakeToggle(LayersPanel, "Heart", "Heart", true, on => SetGroup(false, on));
            MakeToggle(LayersPanel, "Fat", "EAT overlay (display layer)", true, on => SetGroup(true, on));
            MakeToggle(LayersPanel, "CT", "CT on the cut", m_Slice == null || m_Slice.showCt, on => { if (m_Slice != null) m_Slice.showCt = on; });
            MakeToggle(LayersPanel, "Realistic", "Realistic look", m_Look == null || m_Look.CurrentLook == HeartMaterials.Look.Realistic,
                on => { if (m_Look != null) m_Look.CurrentLook = on ? HeartMaterials.Look.Realistic : HeartMaterials.Look.Labelled; });
            Label(LayersPanel, "Wall opacity", 20, k_Muted, TextAnchor.MiddleLeft);
            m_Opacity = MakeSlider(LayersPanel, 0f, 1f, m_Controller != null ? m_Controller.wallOpacity : 1f,
                v => { if (m_Controller != null) m_Controller.wallOpacity = v; });
            Label(LayersPanel, "Structures", 20, k_Muted, TextAnchor.MiddleLeft);
            if (m_Controller != null)
                foreach (var name in m_Controller.Structures.Keys)
                {
                    var n = name;
                    MakeToggle(LayersPanel, n, n, true, on => m_Controller.SetVisible(n, on), 19, 28f);
                }
            MakeButton(LayersPanel, "Reset2", "Reset view", 0f, ResetView);
        }

        void BuildCase()
        {
            CasePanel = Panel("Case", m_Canvas.transform, new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-24f, -24f), 380f, true);
            var c = m_Slice != null ? m_Slice.Case : null;
            m_CaseTitle = Label(CasePanel, c != null ? $"Case {c.case_id}" : "Case", 30, k_Text, TextAnchor.MiddleLeft, FontStyle.Bold);
            if (c == null)
            {
                Label(CasePanel, "No case package. Copy output/case_package/<case id>/ to Assets/StreamingAssets/Cases/.", 19, k_Muted, TextAnchor.UpperLeft);
                return;
            }
            foreach (var m in c.measurements)
            {
                var development = m.status == "in development";
                Label(CasePanel, development || string.IsNullOrEmpty(m.status) ? m.name : $"{m.name}  ·  {m.status}", 18, k_Muted, TextAnchor.MiddleLeft);
                Label(CasePanel, m.text, 25, development ? k_Accent : k_Text, TextAnchor.MiddleLeft, FontStyle.Bold);
            }
            if (c.motion != null)
                Label(CasePanel, $"Motion: {c.motion.kind}, {c.motion.bpm:0} bpm", 19, k_Muted, TextAnchor.MiddleLeft);
            Label(CasePanel, c.notice, 18, k_Muted, TextAnchor.UpperLeft, FontStyle.Italic);
        }

        void BuildSlicePanel()
        {
            SlicePanel = Panel("Slice", m_Canvas.transform, new Vector2(1f, 0f), new Vector2(1f, 0f), new Vector2(-24f, 24f), 360f, true);
            m_SliceText = Label(SlicePanel, "Slice", 24, k_Text, TextAnchor.MiddleLeft, FontStyle.Bold);
            var row = Row(SlicePanel);
            MakeButton(row, "Axial", "Axial", 82f, () => SetOrientation(HeartSlice.Orientation.Axial), 42f);
            MakeButton(row, "Coronal", "Coronal", 98f, () => SetOrientation(HeartSlice.Orientation.Coronal), 42f);
            MakeButton(row, "Sagittal", "Sagittal", 98f, () => SetOrientation(HeartSlice.Orientation.Sagittal), 42f);
            MakeButton(row, "Flip", "Flip", 54f, () => { if (m_Slice != null) m_Slice.flip = !m_Slice.flip; }, 42f);
            m_SlicePosition = MakeSlider(SlicePanel, -1f, 1f, m_Slice != null ? m_Slice.position : 0f,
                v => { if (m_Slice != null) m_Slice.position = v; });
            Label(SlicePanel, "CT in the plane of the cut", 19, k_Muted, TextAnchor.MiddleLeft);
            var view = new GameObject("CT view", typeof(RectTransform), typeof(RawImage), typeof(LayoutElement));
            view.transform.SetParent(SlicePanel, false);
            m_CtView = view.GetComponent<RawImage>();
            m_CtViewSize = view.GetComponent<LayoutElement>();
            m_CtViewSize.preferredWidth = k_CtView;
            m_CtViewSize.preferredHeight = k_CtView;
            SlicePanel.gameObject.SetActive(false);
        }

        void BuildCallout()
        {
            var line = new GameObject("Callout line", typeof(RectTransform), typeof(Image));
            line.transform.SetParent(m_Canvas.transform, false);
            m_CalloutLine = (RectTransform)line.transform;
            m_CalloutLine.anchorMin = m_CalloutLine.anchorMax = new Vector2(0.5f, 0.5f);
            m_CalloutLine.pivot = new Vector2(0f, 0.5f);
            var img = line.GetComponent<Image>();
            img.color = new Color(1f, 1f, 1f, 0.85f);
            img.raycastTarget = false;
            m_Callout = Panel("Callout", m_Canvas.transform, new Vector2(0.5f, 0.5f), new Vector2(0f, 0f), Vector2.zero, 0f, false);
            m_Callout.GetComponent<Image>().raycastTarget = false;
            m_CalloutText = Label(m_Callout, "", 22, k_Text, TextAnchor.MiddleLeft);
            m_CalloutText.raycastTarget = false;
            ShowCallout(false);
        }

        // ── behaviour ─────────────────────────────────────────────────────────
        void SetMode(HeartMouseInput.DragMode mode)
        {
            if (m_Mouse != null) m_Mouse.mode = mode;
        }

        void ToggleSlice()
        {
            if (m_Slice == null) return;
            m_Slice.active = !m_Slice.active;
            SetMode(m_Slice.active ? HeartMouseInput.DragMode.Slice : HeartMouseInput.DragMode.Rotate);
            m_Slice.Apply();
        }

        void SetOrientation(HeartSlice.Orientation o)
        {
            if (m_Slice == null) return;
            m_Slice.orientation = o;
            m_Slice.Apply();
        }

        void ChangeRate(float delta)
        {
            if (m_Controller != null) m_Controller.heartRate = Mathf.Clamp(m_Controller.heartRate + delta, 30f, 150f);
        }

        void ResetView()
        {
            if (m_Manipulator != null) m_Manipulator.ResetView();
        }

        void SetGroup(bool fat, bool on)
        {
            if (m_Controller == null) return;
            foreach (var name in m_Controller.Structures.Keys.ToList())
                if ((name == k_Fat) == fat) m_Controller.SetVisible(name, on);
        }

        void Update()
        {
            if (m_Canvas == null) return;
            if (hoverLabels && m_Picker != null && Application.isPlaying)
            {
                var cam = uiCamera != null ? uiCamera : Camera.main;
                var busy = m_Mouse != null && m_Mouse.Dragging || EventSystem.current != null && EventSystem.current.IsPointerOverGameObject();
                if (cam != null && !busy && m_Picker.Pick(cam.ScreenPointToRay(Input.mousePosition), out var hit)) ShowCallout(hit, cam);
                else ShowCallout(false);
            }
            Refresh();
        }

        /// <summary>Bring every control in line with the heart's state (also after changes made elsewhere).</summary>
        public void Refresh()
        {
            if (m_Stage != null)
            {
                Highlight("Seated", m_Stage.posture == HeartStage.Posture.Seated);
                Highlight("Standing", m_Stage.posture == HeartStage.Posture.Standing);
                Highlight("Desktop", m_Stage.quality == HeartStage.Quality.Desktop);
                Highlight("Headset", m_Stage.quality == HeartStage.Quality.Headset);
                if (m_CaseButton != null && m_Stage.Loading) m_CaseButton.text = "Loading ...";
            }
            if (m_Controller != null)
            {
                m_Play.text = m_Controller.playing ? "Pause" : "Play";
                m_PlayIcon.sprite = HeartIcons.Get(m_Controller.playing ? "pause" : "play");
                m_Rate.text = $"{m_Controller.heartRate:0} bpm";
                m_Opacity.SetValueWithoutNotify(m_Controller.wallOpacity);
                var any = false; var fat = false;
                foreach (var kv in m_Controller.Structures)
                {
                    if (m_Toggles.TryGetValue(kv.Key, out var t)) t.SetIsOnWithoutNotify(kv.Value.enabled);
                    if (kv.Key == k_Fat) fat = kv.Value.enabled; else any |= kv.Value.enabled;
                }
                m_Toggles["Heart"].SetIsOnWithoutNotify(any);
                m_Toggles["Fat"].SetIsOnWithoutNotify(fat);
            }
            if (m_Look != null) m_Toggles["Realistic"].SetIsOnWithoutNotify(m_Look.CurrentLook == HeartMaterials.Look.Realistic);
            var mode = m_Mouse != null ? m_Mouse.mode : HeartMouseInput.DragMode.Rotate;
            Highlight("Rotate", mode == HeartMouseInput.DragMode.Rotate);
            Highlight("Zoom", mode == HeartMouseInput.DragMode.Zoom);
            Highlight("Slice", m_Slice != null && m_Slice.active);
            var slicing = m_Slice != null && m_Slice.active;
            if (SlicePanel.gameObject.activeSelf != slicing) SlicePanel.gameObject.SetActive(slicing);
            if (!slicing) return;
            m_Toggles["CT"].SetIsOnWithoutNotify(m_Slice.showCt);
            m_SlicePosition.SetValueWithoutNotify(m_Slice.position);
            foreach (var o in new[] { HeartSlice.Orientation.Axial, HeartSlice.Orientation.Coronal, HeartSlice.Orientation.Sagittal })
                Highlight(o.ToString(), m_Slice.orientation == o);
            Highlight("Flip", m_Slice.flip);
            m_SliceText.text = $"Slice: {m_Slice.orientation.ToString().ToLowerInvariant()}" + (m_Slice.Case == null ? " (no CT: no case package)" : "");
            var key = $"{m_Slice.orientation} {m_Slice.position:0.000} {m_Slice.windowCenter} {m_Slice.windowWidth}";
            if (key != m_CtKey && m_Slice.Case != null)
            {
                m_CtKey = key;
                m_CtImage = m_Slice.CtImage(m_CtImage);
                m_CtView.texture = m_CtImage;
                m_CtViewSize.preferredHeight = k_CtView * m_CtImage.height / m_CtImage.width;
            }
        }

        public void ShowCallout(bool on)
        {
            if (m_Callout == null) return;
            m_Callout.gameObject.SetActive(on);
            m_CalloutLine.gameObject.SetActive(on);
        }

        /// <summary>Label a structure: its name (and volume) in a box joined by a line to the point on it.</summary>
        public void ShowCallout(HeartPicker.Hit hit, Camera cam)
        {
            var c = m_Slice != null ? m_Slice.Case : null;
            var s = c != null ? c.Structure(hit.structure) : null;
            var text = hit.structure;
            if (s != null) text += $"  ·  {s.volume_ml:0.0} mL" + (s.kind == "display layer" ? "  ·  display layer" : "");
            if (hit.onSliceFace) text += "\non the cut (from the CT labels)";
            m_CalloutText.text = text;
            var canvasRect = (RectTransform)m_Canvas.transform;
            var screen = cam.WorldToScreenPoint(hit.point);
            var eventCam = m_Canvas.renderMode == RenderMode.ScreenSpaceOverlay ? null : m_Canvas.worldCamera;
            if (screen.z < 0f || !RectTransformUtility.ScreenPointToLocalPointInRectangle(canvasRect, screen, eventCam, out var p))
            {
                ShowCallout(false);
                return;
            }
            var box = p + new Vector2(90f, 60f);
            m_Callout.anchoredPosition = box;
            var d = box - p;
            m_CalloutLine.anchoredPosition = p;
            m_CalloutLine.sizeDelta = new Vector2(d.magnitude, 2.5f);
            m_CalloutLine.localRotation = Quaternion.Euler(0f, 0f, Mathf.Atan2(d.y, d.x) * Mathf.Rad2Deg);
            ShowCallout(true);
        }

        // ── widgets ───────────────────────────────────────────────────────────
        RectTransform Panel(string name, Transform parent, Vector2 anchor, Vector2 pivot, Vector2 position, float width, bool vertical)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image));
            var rt = (RectTransform)go.transform;
            rt.SetParent(parent, false);
            rt.anchorMin = rt.anchorMax = anchor;
            rt.pivot = pivot;
            rt.anchoredPosition = position;
            rt.sizeDelta = new Vector2(width, 100f);
            var img = go.GetComponent<Image>();
            img.sprite = m_Round;
            img.type = Image.Type.Sliced;
            img.color = k_Panel;
            m_Panels.Add(img);
            HorizontalOrVerticalLayoutGroup layout = vertical ? (HorizontalOrVerticalLayoutGroup)go.AddComponent<VerticalLayoutGroup>() : go.AddComponent<HorizontalLayoutGroup>();
            layout.padding = new RectOffset(18, 18, 14, 14);
            layout.spacing = vertical ? 6f : 8f;
            layout.childAlignment = vertical ? TextAnchor.UpperLeft : TextAnchor.MiddleCenter;
            layout.childControlWidth = layout.childControlHeight = true;
            layout.childForceExpandWidth = vertical;
            layout.childForceExpandHeight = false;
            var fit = go.AddComponent<ContentSizeFitter>();
            fit.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            if (!vertical || width <= 0f) fit.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
            return rt;
        }

        RectTransform Row(Transform parent)
        {
            var go = new GameObject("Row", typeof(RectTransform), typeof(HorizontalLayoutGroup));
            go.transform.SetParent(parent, false);
            var h = go.GetComponent<HorizontalLayoutGroup>();
            h.spacing = 6f;
            h.childControlWidth = h.childControlHeight = true;
            h.childForceExpandWidth = h.childForceExpandHeight = false;
            return (RectTransform)go.transform;
        }

        void Spacer(Transform parent, float width)
        {
            var go = new GameObject("Spacer", typeof(RectTransform), typeof(LayoutElement));
            go.transform.SetParent(parent, false);
            go.GetComponent<LayoutElement>().preferredWidth = width;
        }

        Text Label(Transform parent, string text, int size, Color colour, TextAnchor align, FontStyle style = FontStyle.Normal)
        {
            var go = new GameObject("Text", typeof(RectTransform), typeof(Text));
            go.transform.SetParent(parent, false);
            var t = go.GetComponent<Text>();
            t.font = m_Font;
            t.fontSize = size;
            t.fontStyle = style;
            t.color = colour;
            t.alignment = align;
            t.text = text;
            t.horizontalOverflow = HorizontalWrapMode.Wrap;
            t.verticalOverflow = VerticalWrapMode.Overflow;
            t.raycastTarget = false;
            return t;
        }

        Image Icon(Transform parent, string name, float size)
        {
            var go = new GameObject("Icon", typeof(RectTransform), typeof(Image), typeof(LayoutElement));
            go.transform.SetParent(parent, false);
            var le = go.GetComponent<LayoutElement>();
            le.preferredWidth = le.preferredHeight = le.minWidth = le.minHeight = size;
            var img = go.GetComponent<Image>();
            img.sprite = HeartIcons.Get(name);
            img.preserveAspect = true;
            img.raycastTarget = false;
            img.color = k_Text;
            return img;
        }

        Button MakeButton(Transform parent, string key, string text, float width, Action onClick, float height = 52f, string icon = null)
        {
            var go = new GameObject(key, typeof(RectTransform), typeof(Image), typeof(Button), typeof(LayoutElement));
            go.transform.SetParent(parent, false);
            var img = go.GetComponent<Image>();
            img.sprite = m_Round;
            img.type = Image.Type.Sliced;
            var b = go.GetComponent<Button>();
            b.targetGraphic = img;
            b.colors = Colours(false);
            var le = go.GetComponent<LayoutElement>();
            if (width > 0f) le.preferredWidth = width;
            le.preferredHeight = height;
            if (icon != null)
            {
                // icon and label side by side
                var row = go.AddComponent<HorizontalLayoutGroup>();
                row.padding = new RectOffset(12, 14, 4, 4);
                row.spacing = 8f;
                row.childAlignment = TextAnchor.MiddleCenter;
                row.childControlWidth = row.childControlHeight = true;
                row.childForceExpandWidth = row.childForceExpandHeight = false;
                Icon(go.transform, icon, height * 0.5f);
                Label(go.transform, text, 22, k_Text, TextAnchor.MiddleLeft);
            }
            else
            {
                var t = Label(go.transform, text, 22, k_Text, TextAnchor.MiddleCenter);
                var tr = (RectTransform)t.transform;
                tr.anchorMin = Vector2.zero; tr.anchorMax = Vector2.one;
                tr.offsetMin = tr.offsetMax = Vector2.zero;
            }
            b.onClick.AddListener(() => onClick());
            m_Buttons[key] = b;
            return b;
        }

        Toggle MakeToggle(Transform parent, string key, string text, bool on, Action<bool> changed, int size = 22, float height = 36f)
        {
            var go = new GameObject("Toggle " + key, typeof(RectTransform), typeof(Toggle), typeof(LayoutElement));
            go.transform.SetParent(parent, false);
            go.GetComponent<LayoutElement>().preferredHeight = height;
            var box = new GameObject("Box", typeof(RectTransform), typeof(Image));
            box.transform.SetParent(go.transform, false);
            var br = (RectTransform)box.transform;
            br.anchorMin = br.anchorMax = new Vector2(0f, 0.5f);
            br.pivot = new Vector2(0f, 0.5f);
            br.sizeDelta = new Vector2(height - 10f, height - 10f);
            var boxImg = box.GetComponent<Image>();
            boxImg.sprite = m_Round; boxImg.type = Image.Type.Sliced; boxImg.color = new Color(1f, 1f, 1f, 0.16f);
            var check = new GameObject("Check", typeof(RectTransform), typeof(Image));
            check.transform.SetParent(box.transform, false);
            var cr = (RectTransform)check.transform;
            cr.anchorMin = Vector2.zero; cr.anchorMax = Vector2.one;
            cr.offsetMin = new Vector2(5f, 5f); cr.offsetMax = new Vector2(-5f, -5f);
            var checkImg = check.GetComponent<Image>();
            checkImg.sprite = m_Round; checkImg.type = Image.Type.Sliced; checkImg.color = k_Accent;
            var label = Label(go.transform, text, size, k_Text, TextAnchor.MiddleLeft);
            var lr = (RectTransform)label.transform;
            lr.anchorMin = Vector2.zero; lr.anchorMax = Vector2.one;
            lr.offsetMin = new Vector2(height + 2f, 0f); lr.offsetMax = Vector2.zero;
            var t = go.GetComponent<Toggle>();
            t.targetGraphic = boxImg;
            t.graphic = checkImg;
            t.isOn = on;
            t.onValueChanged.AddListener(v => changed(v));
            m_Toggles[key] = t;
            return t;
        }

        Slider MakeSlider(Transform parent, float min, float max, float value, Action<float> changed)
        {
            var go = new GameObject("Slider", typeof(RectTransform), typeof(Slider), typeof(LayoutElement));
            go.transform.SetParent(parent, false);
            go.GetComponent<LayoutElement>().preferredHeight = 34f;
            RectTransform Child(string name, Transform p, Vector2 aMin, Vector2 aMax, Vector2 offMin, Vector2 offMax)
            {
                var c = new GameObject(name, typeof(RectTransform));
                c.transform.SetParent(p, false);
                var r = (RectTransform)c.transform;
                r.anchorMin = aMin; r.anchorMax = aMax; r.offsetMin = offMin; r.offsetMax = offMax;
                return r;
            }
            var bg = Child("Background", go.transform, new Vector2(0f, 0.38f), new Vector2(1f, 0.62f), Vector2.zero, Vector2.zero).gameObject.AddComponent<Image>();
            bg.sprite = m_Round; bg.type = Image.Type.Sliced; bg.color = new Color(1f, 1f, 1f, 0.16f);
            var fillArea = Child("Fill Area", go.transform, new Vector2(0f, 0.38f), new Vector2(1f, 0.62f), new Vector2(4f, 0f), new Vector2(-4f, 0f));
            var fill = Child("Fill", fillArea, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero).gameObject.AddComponent<Image>();
            fill.sprite = m_Round; fill.type = Image.Type.Sliced; fill.color = k_Accent;
            var handleArea = Child("Handle Slide Area", go.transform, Vector2.zero, Vector2.one, new Vector2(12f, 0f), new Vector2(-12f, 0f));
            var handleRect = Child("Handle", handleArea, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            handleRect.sizeDelta = new Vector2(26f, 0f);
            var handle = handleRect.gameObject.AddComponent<Image>();
            handle.sprite = m_Circle; handle.color = Color.white;
            var s = go.GetComponent<Slider>();
            s.fillRect = fill.rectTransform;
            s.handleRect = handleRect;
            s.targetGraphic = handle;
            s.direction = Slider.Direction.LeftToRight;
            s.minValue = min; s.maxValue = max; s.value = value;
            s.onValueChanged.AddListener(v => changed(v));
            return s;
        }

        void Highlight(string key, bool on)
        {
            if (m_Buttons.TryGetValue(key, out var b)) b.colors = Colours(on);
        }

        static ColorBlock Colours(bool on)
        {
            var c = ColorBlock.defaultColorBlock;
            c.normalColor = on ? new Color(k_Accent.r, k_Accent.g, k_Accent.b, 0.55f) : new Color(1f, 1f, 1f, 0.10f);
            c.highlightedColor = on ? new Color(k_Accent.r, k_Accent.g, k_Accent.b, 0.7f) : new Color(1f, 1f, 1f, 0.22f);
            c.pressedColor = new Color(k_Accent.r, k_Accent.g, k_Accent.b, 0.85f);
            c.selectedColor = c.normalColor;
            c.colorMultiplier = 1f;
            return c;
        }

        /// <summary>White rounded square (alpha anti-aliased), for 9-sliced panels and buttons; a circle when radius = size / 2.</summary>
        static Texture2D RoundedTexture(int size, float radius)
        {
            var t = new Texture2D(size, size, TextureFormat.RGBA32, false) { hideFlags = HideFlags.DontSave, wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[size * size];
            for (var y = 0; y < size; y++)
            for (var x = 0; x < size; x++)
            {
                var dx = Mathf.Max(radius - (x + 0.5f), 0f, x + 0.5f - (size - radius));
                var dy = Mathf.Max(radius - (y + 0.5f), 0f, y + 0.5f - (size - radius));
                var a = Mathf.Clamp01(radius - Mathf.Sqrt(dx * dx + dy * dy) + 0.5f);
                px[y * size + x] = new Color32(255, 255, 255, (byte)Mathf.RoundToInt(a * 255f));
            }
            t.SetPixels32(px);
            t.Apply(false);
            return t;
        }

        void Clear()
        {
            m_Buttons.Clear();
            m_Toggles.Clear();
            m_Panels.Clear();
            foreach (var o in new UnityEngine.Object[] { m_Canvas != null ? m_Canvas.gameObject : null, m_Round, m_Circle, m_RoundTex, m_CircleTex, m_CtImage, m_Material, m_GlassMaterial })
                if (o != null) { if (Application.isPlaying) Destroy(o); else DestroyImmediate(o); }
            m_Canvas = null;
        }
    }
}
