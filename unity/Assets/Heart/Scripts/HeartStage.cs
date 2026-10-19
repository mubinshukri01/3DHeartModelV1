// UTM-Claude Research Project | rohayanti@utm.my
// The stage the heart sits on: the desk in the demo scene, the real table in mixed reality. It holds what
// concerns the whole experience rather than one heart:
//   * quality: Desktop (the full model, frosted-glass panels on the screen) or Headset (the case package's
//     light model, panels floating in front of the viewer without blur, an opaque wall by default)
//   * comfort: seated or standing; Recentre puts the stage in front of the viewer at a comfortable height
//     (on the desktop it puts it back where it started)
//   * cases: the case packages in StreamingAssets/Cases, and loading one at runtime (model, CT, case card)
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GLTFast;
using UnityEngine;

namespace UTM.Heart
{
    /// <summary>Put this on the heart's parent (the stage), which also holds the desk.</summary>
    [DisallowMultipleComponent]
    public class HeartStage : MonoBehaviour
    {
        public enum Quality { Desktop, Headset }
        public enum Posture { Seated, Standing }

        [Tooltip("Desktop: full model, glass panels on the screen. Headset: light model, panels in front of the viewer.")]
        public Quality quality = Quality.Desktop;
        public Posture posture = Posture.Seated;
        [Tooltip("Desk top below eye level (m), seated and standing")]
        public float seatedDrop = 0.40f, standingDrop = 0.55f;
        [Tooltip("Distance from the viewer's eyes to the heart (m), across the floor")]
        public float reach = 0.55f;
        public KeyCode recentreKey = KeyCode.C;
        [Tooltip("The viewer (default: the main camera)")]
        public Camera viewer;

        [Header("Set-up of hearts loaded at runtime (Tools > Heart > Create demo scene fills these)")]
        public Material ventricleMuscle;
        public Material atrialMuscle, wall, blood, oxygenated, deoxygenated, fat;
        public Shader contactShadowShader, sliceCapShader, uiGlassShader;

        Vector3 m_HomePosition;
        Quaternion m_HomeRotation;
        bool m_HomeSet;

        /// <summary>The heart on the stage now (its root, with HeartController).</summary>
        public GameObject Heart { get; private set; }
        public string CaseId { get; private set; }
        public bool Loading { get; private set; }
        public string LastError { get; private set; }

        public static string CasesFolder => Path.Combine(Application.streamingAssetsPath, "Cases");

        /// <summary>Case packages in StreamingAssets/Cases (folders with a case.json), by name.</summary>
        public static string[] AvailableCases() =>
            Directory.Exists(CasesFolder)
                ? Directory.GetDirectories(CasesFolder).Where(d => File.Exists(Path.Combine(d, "case.json")))
                    .Select(Path.GetFileName).OrderBy(n => n, StringComparer.Ordinal).ToArray()
                : new string[0];

        void Awake()
        {
            RememberHome();
            FindHeart();
        }

        /// <summary>Where Recentre puts the stage back on the desktop: where it is now (called on Awake).</summary>
        public void RememberHome()
        {
            m_HomePosition = transform.localPosition;
            m_HomeRotation = transform.localRotation;
            m_HomeSet = true;
        }

        void Start()
        {
            if (quality == Quality.Headset) ApplyQuality();
        }

        void Update()
        {
            if (Input.GetKeyDown(recentreKey)) Recentre();
        }

        public void FindHeart()
        {
            var ctrl = GetComponentInChildren<HeartController>(true);
            Heart = ctrl != null ? ctrl.gameObject : null;
            var slice = Heart != null ? Heart.GetComponent<HeartSlice>() : null;
            CaseId = slice != null ? slice.caseId : null;
        }

        // ── comfort ───────────────────────────────────────────────────────────
        /// <summary>
        /// Headset: put the stage in front of the viewer, the desk top <see cref="seatedDrop"/> or
        /// <see cref="standingDrop"/> below the eyes and the heart <see cref="reach"/> away, its front facing them.
        /// Desktop (the camera does not move): put the stage back where it started.
        /// </summary>
        public void Recentre()
        {
            if (!m_HomeSet) RememberHome();                                    // edit mode: no Awake
            if (Heart == null) FindHeart();
            var cam = viewer != null ? viewer : Camera.main;
            if (quality == Quality.Desktop || cam == null)
            {
                transform.localPosition = m_HomePosition;
                transform.localRotation = m_HomeRotation;
                return;
            }
            var eye = cam.transform.position;
            var ahead = Vector3.ProjectOnPlane(cam.transform.forward, Vector3.up);
            ahead = ahead.sqrMagnitude > 1e-6f ? ahead.normalized : Vector3.forward;
            var target = eye + ahead * reach;
            target.y = eye.y - (posture == Posture.Seated ? seatedDrop : standingDrop);
            var rotation = Quaternion.LookRotation(-ahead, Vector3.up);          // the heart's front (+Z) faces the viewer
            var manip = Heart != null ? Heart.GetComponent<HeartManipulator>() : null;
            var under = manip != null && manip.IsReady ? new Vector3(manip.TargetCentre.x, manip.DeskHeight, manip.TargetCentre.z) : Vector3.zero;
            transform.SetPositionAndRotation(target - rotation * Vector3.Scale(transform.lossyScale, under), rotation);
        }

        public void SetPosture(Posture p)
        {
            posture = p;
            if (quality == Quality.Headset) Recentre();
        }

        // ── quality ───────────────────────────────────────────────────────────
        /// <summary>Switch quality: the panels' mode, and (when the model must change) reload the case.</summary>
        public async Task SetQuality(Quality q)
        {
            if (q == quality) return;
            quality = q;
            if (CaseId != null && Heart != null) await LoadCase(CaseId);
            else ApplyQuality();
        }

        void ApplyQuality()
        {
            if (Heart == null) return;
            var ui = Heart.GetComponent<HeartUI>();
            var ctrl = Heart.GetComponent<HeartController>();
            if (ui != null)
            {
                ui.glass = quality == Quality.Desktop;
                ui.SetRenderMode(quality == Quality.Headset ? RenderMode.WorldSpace : RenderMode.ScreenSpaceOverlay);
            }
            if (ctrl != null && quality == Quality.Headset) ctrl.wallOpacity = 1f;   // less overdraw in the headset
        }

        // ── cases ─────────────────────────────────────────────────────────────
        /// <summary>
        /// Load a case package at runtime: its model (the light one for Headset quality), CT and case card, set up
        /// like the demo scene's heart, resting on the same desk; then it replaces the current heart.
        /// </summary>
        public async Task<bool> LoadCase(string caseId)
        {
            if (Loading) return false;
            Loading = true;
            LastError = null;
            GameObject holder = null;
            try
            {
                var folder = Path.Combine(CasesFolder, caseId);
                var card = HeartCase.Load(folder);
                var file = quality == Quality.Headset && !string.IsNullOrEmpty(card.model.light_file) ? card.model.light_file : card.model.file;
                var gltf = new GltfImport();
                var settings = new ImportSettings { AnimationMethod = AnimationMethod.Legacy };
                if (!await gltf.Load(File.ReadAllBytes(Path.Combine(folder, file)), new Uri(Path.Combine(folder, file)), settings))
                    throw new InvalidDataException($"{file} could not be loaded");
                holder = new GameObject("Loading " + caseId);
                holder.transform.SetParent(transform, false);
                holder.SetActive(false);                                       // nothing runs before it is set up
                if (!await gltf.InstantiateMainSceneAsync(holder.transform))
                    throw new InvalidDataException($"{file} could not be instantiated");
                var heart = holder.transform.GetChild(0).gameObject;          // glTFast's scene object, with the Animation
                heart.name = "Heart";
                heart.transform.SetParent(transform, false);
                heart.SetActive(false);
                Destroy(holder);
                holder = null;
                PlaceOnDesk(heart);
                SetUp(heart, caseId, card.motion != null && card.motion.bpm > 0f ? card.motion.bpm : 72f);
                var old = Heart;
                Heart = heart;
                CaseId = caseId;
                if (old != null) Destroy(old);
                heart.SetActive(true);                                         // OnEnable / Start set everything up
                ApplyQuality();
                return true;
            }
            catch (Exception e)
            {
                LastError = e.Message;
                Debug.LogError($"HeartStage: case {caseId}: {e.Message}", this);
                if (holder != null) Destroy(holder);
                return false;
            }
            finally
            {
                Loading = false;
            }
        }

        /// <summary>Centre the new heart over the stage, its lowest point on the current heart's desk.</summary>
        void PlaceOnDesk(GameObject heart)
        {
            var manip = Heart != null ? Heart.GetComponent<HeartManipulator>() : null;
            var desk = manip != null && manip.IsReady ? manip.DeskHeight + manip.gap : 0f;
            heart.transform.localRotation = Quaternion.identity;
            heart.transform.localScale = Vector3.one;
            heart.transform.localPosition = Vector3.zero;
            var points = HeartContactShadow.Vertices(heart.transform, transform);
            if (points.Count == 0) return;
            var b = new Bounds(points[0], Vector3.zero);
            foreach (var p in points) b.Encapsulate(p);
            heart.transform.localPosition = new Vector3(-b.center.x, desk - b.min.y, -b.center.z);
        }

        /// <summary>The demo scene's set of components on a heart loaded at runtime.</summary>
        void SetUp(GameObject heart, string caseId, float bpm)
        {
            var ctrl = heart.AddComponent<HeartController>();                    // plays glTFast's legacy clip
            ctrl.clipBpm = bpm;
            ctrl.heartRate = bpm;
            var look = heart.AddComponent<HeartMaterials>();
            look.ventricleMuscle = ventricleMuscle; look.atrialMuscle = atrialMuscle; look.wall = wall;
            look.blood = blood; look.oxygenated = oxygenated; look.deoxygenated = deoxygenated; look.fat = fat;
            heart.AddComponent<HeartContactShadow>().shader = contactShadowShader;
            heart.AddComponent<HeartManipulator>();
            heart.AddComponent<HeartMouseInput>();
            var slice = heart.AddComponent<HeartSlice>();
            slice.capShader = sliceCapShader;
            slice.caseId = caseId;
            heart.AddComponent<HeartPicker>();
            heart.AddComponent<HeartUI>().glassShader = uiGlassShader;
        }
    }
}
