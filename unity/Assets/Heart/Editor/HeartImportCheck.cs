// UTM-Claude Research Project | rohayanti@utm.my
// Checks the Unity import of heart_animated.glb (glTFast), HeartController, the tissue materials
// (HeartMaterials), the contact shadow (HeartContactShadow) and rotate / zoom / move / reset
// (HeartManipulator), and builds a demo scene.
// Command line (exit code 0 = all checks passed, report in Logs/heart_import_check.txt):
//   Unity.exe -batchmode -nographics -projectPath <project>
//             -executeMethod UTM.Heart.EditorTools.HeartImportCheck.Run -logFile <log>
// In the Editor: Tools > Heart > Run import check / Create demo scene.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace UTM.Heart.EditorTools
{
    public static class HeartImportCheck
    {
        public const string GlbPath = "Assets/Heart/heart_animated.glb";
        public const string ScenePath = "Assets/Heart/HeartDemo.unity";
        public const string MaterialsFolder = "Assets/Heart/Materials";
        const string ReportPath = "Logs/heart_import_check.txt";

        static readonly string[] k_Structures =
        {
            "Myocardium", "LV blood pool", "LA blood pool", "Aorta", "Right atrium", "Right ventricle",
            "Pulmonary artery", "Superior vena cava", "Epicardial fat (EAT)",
        };
        // present when vessels.py (TotalSegmentator) has run before build_heart_model.py
        static readonly string[] k_Optional = { "Inferior vena cava", "Pulmonary veins", "Descending aorta", "Coronary arteries" };
        static readonly string[] k_Targets = { "ventricular_systole", "atrial_systole", "great_artery_distension" };

        // From output/3d_models/animation_info.json (build_heart_model.py, 72 bpm): LV blood pool volume
        // at phase 0 (rest pose) and at 0.325 of the R-R interval (a keyframe, ventricular weight 0.99)
        // (shared-surface meshing since 9 Oct 2026; before: 31.60 / 14.28 mL, shrunk by a smoothing bug;
        // blended motion since 10 Oct 2026: systole 14.98 -> 15.14 mL; deep-learning chambers since 12 Oct 2026,
        // TotalSegmentator heartchambers_highres: 52.32 -> 22.77 mL)
        const float k_LvRestMl = 52.32f, k_LvSystoleMl = 22.77f, k_SystolePhase = 0.325f;
        const string k_ShadowShader = "Heart/Contact Shadow";
        public const string SliceCapShader = "Heart/Slice Cap";
        public const string GlassShader = "Heart/UI Glass";
        const float k_ViewPitch = 20f;      // demo camera looks down on the desk, like someone seated at it

        // realistic look: material asset, shader, default colour (created once; later edits in the Inspector are kept)
        static readonly (string file, string shader, Color color)[] k_Materials =
        {
            ("Ventricle muscle", "Heart/Muscle", new Color(0.48f, 0.10f, 0.09f, 1f)),
            ("Atrial muscle", "Heart/Muscle", new Color(0.40f, 0.09f, 0.12f, 1f)),
            ("Wall", "Heart/Muscle (see-through)", new Color(0.48f, 0.10f, 0.09f, 1f)),
            ("Blood", "Heart/Vessel", new Color(0.38f, 0.02f, 0.03f, 1f)),
            ("Oxygenated vessel", "Heart/Vessel", new Color(0.66f, 0.10f, 0.09f, 1f)),
            ("Deoxygenated vessel", "Heart/Vessel", new Color(0.16f, 0.24f, 0.55f, 1f)),
            ("Fat", "Heart/Fat", new Color(0.96f, 0.78f, 0.38f, 0.72f)),
        };

        [MenuItem("Tools/Heart/Run import check")]
        public static void RunFromMenu() => Debug.Log(Check(out _));

        [MenuItem("Tools/Heart/Create demo scene")]
        public static void CreateDemoSceneFromMenu() => CreateDemoScene();

        /// <summary>Command-line entry point: writes the report and exits with 0 (pass) or 1 (fail).</summary>
        public static void Run()
        {
            string report;
            var ok = false;
            try { report = Check(out ok); }
            catch (Exception e) { report = "EXCEPTION " + e; }
            Directory.CreateDirectory(Path.GetDirectoryName(ReportPath));
            File.WriteAllText(ReportPath, report);
            Debug.Log(report);
            EditorApplication.Exit(ok ? 0 : 1);
        }

        static string Check(out bool ok)
        {
            var lines = new List<string>();
            var fails = 0;
            void Expect(bool condition, string what)
            {
                lines.Add((condition ? "PASS " : "FAIL ") + what);
                if (!condition) fails++;
            }

            // ── import ──────────────────────────────────────────────────────
            AssetDatabase.ImportAsset(GlbPath, ImportAssetOptions.ForceUpdate);
            var root = AssetDatabase.LoadAssetAtPath<GameObject>(GlbPath);
            Expect(root != null, $"{GlbPath} imports as a GameObject");
            if (root == null) return Finish(lines, fails, out ok);
            var clip = AssetDatabase.LoadAllAssetsAtPath(GlbPath).OfType<AnimationClip>()
                .FirstOrDefault(c => c.name == "CardiacCycle");
            Expect(clip != null, "AnimationClip 'CardiacCycle' imported");
            Expect(root.GetComponent<Animator>() != null, "root has an Animator (glTFast Mecanim import)");
            if (clip != null)
            {
                Expect(Mathf.Abs(clip.length - 60f / 72f) < 0.01f, $"clip length {clip.length:0.000} s (72 bpm = 0.833 s)");
                Expect(AnimationUtility.GetAnimationClipSettings(clip).loopTime, "clip loops");
                var bindings = AnimationUtility.GetCurveBindings(clip);
                var meshCount = root.GetComponentsInChildren<SkinnedMeshRenderer>(true).Length;
                Expect(bindings.Length == meshCount * k_Targets.Length && bindings.All(b => b.propertyName.StartsWith("blendShape.")),
                    $"{bindings.Length} blend-shape curves");
            }

            // ── meshes ──────────────────────────────────────────────────────
            var smrs = root.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            var present = smrs.Select(r => r.name).ToArray();
            var extra = present.Except(k_Structures).ToArray();
            Expect(k_Structures.All(present.Contains) && extra.All(k_Optional.Contains),
                $"{smrs.Length} skinned mesh renderers: the {k_Structures.Length} required + [{string.Join(", ", extra)}]");
            var centres = new Dictionary<string, Vector3>();
            var bounds = new Bounds();
            var triangles = 0;
            foreach (var s in k_Structures.Concat(extra))
            {
                var r = smrs.FirstOrDefault(x => x.name == s);
                Expect(r != null, $"structure '{s}' present");
                if (r == null) continue;
                var m = r.sharedMesh;
                var names = Enumerable.Range(0, m.blendShapeCount).Select(m.GetBlendShapeName).ToArray();
                Expect(names.SequenceEqual(k_Targets),
                    $"'{s}': blend shapes [{string.Join(", ", names)}], {m.vertexCount} vertices, {m.triangles.Length / 3} triangles");
                centres[s] = m.bounds.center;
                if (bounds.size == Vector3.zero) bounds = m.bounds; else bounds.Encapsulate(m.bounds);
                triangles += m.triangles.Length / 3;
            }
            Expect(bounds.size.y > 0.06f && bounds.size.y < 0.25f,
                $"real-world size {bounds.size.x * 100:0.0} x {bounds.size.y * 100:0.0} x {bounds.size.z * 100:0.0} cm (1 unit = 1 m), {triangles} triangles");
            if (k_Structures.All(centres.ContainsKey))
            {
                Expect(centres["Aorta"].y > centres["LV blood pool"].y, "orientation: aorta above the LV (Unity +Y = superior)");
                Expect(centres["LV blood pool"].x < centres["Right ventricle"].x, "orientation: LV on the patient's left = Unity -X (glTFast mirrors X)");
                Expect(centres["Right ventricle"].z > centres["LA blood pool"].z, "orientation: RV anterior to the LA (Unity +Z = anterior)");
                if (centres.ContainsKey("Descending aorta"))
                    Expect(centres["Descending aorta"].z < centres["LA blood pool"].z, "orientation: descending aorta behind the LA");
                if (centres.ContainsKey("Inferior vena cava"))
                    Expect(centres["Inferior vena cava"].y < centres["Right atrium"].y, "orientation: IVC below the RA");
            }
            var wall = smrs.FirstOrDefault(x => x.name == "Myocardium");
            Expect(wall != null && wall.sharedMaterial.renderQueue >= (int)RenderQueue.Transparent,
                $"myocardium material is transparent (render queue {(wall != null ? wall.sharedMaterial.renderQueue : -1)}, shader {(wall != null ? wall.sharedMaterial.shader.name : "-")})");

            // ── tissue shaders ──────────────────────────────────────────────
            foreach (var name in k_Materials.Select(m => m.shader).Distinct().Append(k_ShadowShader).Append(SliceCapShader).Append(GlassShader))
            {
                var shader = Shader.Find(name);
                var errors = shader == null ? -1 : ShaderUtil.GetShaderMessages(shader).Count(m => m.severity == UnityEditor.Rendering.ShaderCompilerMessageSeverity.Error);
                Expect(shader != null && errors == 0 && !ShaderUtil.ShaderHasError(shader), $"shader '{name}' compiles ({errors} errors)");
            }

            // ── animation and HeartController ───────────────────────────────
            if (clip != null && smrs.Length > 0)
            {
                var inst = UnityEngine.Object.Instantiate(root);
                try { CheckController(inst, clip, Expect); }
                finally { UnityEngine.Object.DestroyImmediate(inst); }
            }

            // ── demo scene ──────────────────────────────────────────────────
            if (clip != null) Expect(CreateDemoScene() && File.Exists(ScenePath), $"demo scene saved: {ScenePath}");
            return Finish(lines, fails, out ok);
        }

        static void CheckController(GameObject inst, AnimationClip clip, Action<bool, string> expect)
        {
            var lv = inst.GetComponentsInChildren<SkinnedMeshRenderer>().First(r => r.name == "LV blood pool");
            var ctrl = inst.AddComponent<HeartController>();   // edit mode: OnEnable does not run, so start by hand
            ctrl.cardiacCycle = clip;
            ctrl.showDebugPanel = false;
            ctrl.Play();
            try
            {
                ctrl.SetPhase(0f);
                var rest = LvVolumeMl(lv);
                var w0 = lv.GetBlendShapeWeight(0);
                ctrl.SetPhase(k_SystolePhase);
                var sys = LvVolumeMl(lv);
                var w1 = lv.GetBlendShapeWeight(0);
                expect(Mathf.Abs(w0) < 1e-3f && Mathf.Abs(w1 - 0.99f) < 0.02f,
                    $"ventricular_systole weight {w0:0.000} at rest, {w1:0.000} at {k_SystolePhase:0.000} R-R (glTF 0..1 scale kept)");
                expect(Mathf.Abs(rest / k_LvRestMl - 1f) < 0.03f && Mathf.Abs(sys / k_LvSystoleMl - 1f) < 0.03f,
                    $"LV blood pool in Unity {rest:0.00} -> {sys:0.00} mL (Python pipeline {k_LvRestMl:0.00} -> {k_LvSystoleMl:0.00} mL)");

                ctrl.heartRate = 144f;
                ctrl.Apply();
                ctrl.SetPhase(0f);
                ctrl.Advance(0.1f);
                var expected = 0.1f * 2f / clip.length;
                expect(Math.Abs(ctrl.Speed - 2.0) < 1e-6 && Mathf.Abs(ctrl.Phase - expected) < 1e-3f,
                    $"heart rate 144 bpm -> speed {ctrl.Speed:0.00}, phase after 0.1 s {ctrl.Phase:0.000} (expected {expected:0.000})");
                ctrl.playing = false;
                ctrl.Apply();
                expect(ctrl.Speed == 0.0, "pause -> speed 0");

                ctrl.SetPhase(0.2f);
                expect(ctrl.PhaseName == "Ejection (systole)", $"phase name at 20 % R-R: {ctrl.PhaseName}");

                ctrl.wallOpacity = 0.2f;
                ctrl.Apply();
                var block = new MaterialPropertyBlock();
                ctrl.Structures["Myocardium"].GetPropertyBlock(block);
                var alpha = block.GetColor(Shader.PropertyToID("baseColorFactor")).a;
                expect(Mathf.Abs(alpha - 0.2f) < 1e-4f, $"wall opacity 0.2 -> myocardium alpha {alpha:0.00} (material asset unchanged)");

                ctrl.SetVisible("Aorta", false);
                expect(!ctrl.Structures["Aorta"].enabled && ctrl.Structures.Count == inst.GetComponentsInChildren<SkinnedMeshRenderer>(true).Length,
                    $"show/hide by name ({ctrl.Structures.Count} structures)");
                ctrl.SetVisible("Aorta", true);
                CheckMaterials(inst, ctrl, expect);
                CheckContactShadow(inst, expect);
                CheckManipulator(inst, expect);
            }
            finally { ctrl.Stop(); }
        }

        static void CheckMaterials(GameObject inst, HeartController ctrl, Action<bool, string> expect)
        {
            var renderers = inst.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            var gltf = renderers.ToDictionary(r => r, r => r.sharedMaterial);
            var look = AddMaterials(inst);
            look.CurrentLook = HeartMaterials.Look.Realistic;
            var mapped = renderers.Where(r => look.MaterialFor(r.name) != null).ToArray();
            expect(mapped.Length == renderers.Length && mapped.All(r => r.sharedMaterial == look.MaterialFor(r.name)),
                $"realistic look: {mapped.Length}/{renderers.Length} structures get tissue materials " +
                $"({string.Join(", ", mapped.Select(r => r.sharedMaterial.shader.name).Distinct())})");
            ctrl.wallOpacity = 0.3f;
            ctrl.Apply();
            var block = new MaterialPropertyBlock();
            ctrl.Structures["Myocardium"].GetPropertyBlock(block);
            var c = block.GetColor(Shader.PropertyToID("baseColorFactor"));
            var wallColour = look.wall.GetColor("baseColorFactor");
            expect(Mathf.Abs(c.a - 0.3f) < 1e-4f && Mathf.Abs(c.r - wallColour.r) < 1e-4f,
                $"wall opacity 0.3 with the see-through muscle: alpha {c.a:0.00}, colour kept");
            expect(look.wall.renderQueue >= (int)RenderQueue.Transparent && look.fat.renderQueue > look.wall.renderQueue,
                $"render order: opaque tissue, then the wall ({look.wall.renderQueue}), then the fat ({look.fat.renderQueue})");
            look.CurrentLook = HeartMaterials.Look.Labelled;
            var colour = Shader.PropertyToID("baseColorFactor");
            expect(renderers.All(r => r.sharedMaterial.shader.name.StartsWith("Heart/")
                                      && r.sharedMaterial.GetColor(colour) == gltf[r].GetColor(colour)),
                "labelled look: the GLB's colours, on the tissue shaders (so the slice tool cuts it too)");
            look.StoreRestPositions();
            var lv = renderers.First(r => r.name == "LV blood pool").sharedMesh;
            var uv = new List<Vector3>();
            lv.GetUVs(HeartMaterials.RestUvChannel, uv);
            expect(uv.Count == lv.vertexCount && uv.SequenceEqual(lv.vertices) && lv.blendShapeCount == k_Targets.Length,
                $"rest positions in UV{HeartMaterials.RestUvChannel} ({uv.Count} vertices), blend shapes kept");
            UnityEngine.Object.DestroyImmediate(look);
        }

        static void CheckContactShadow(GameObject inst, Action<bool, string> expect)
        {
            var shadow = inst.AddComponent<HeartContactShadow>();
            shadow.shader = Shader.Find(k_ShadowShader);
            shadow.Build();
            var lowest = HeartContactShadow.LocalVertices(inst.transform).Min(p => p.y);
            var a = shadow.Texture.GetRawTextureData();
            var n = HeartContactShadow.Resolution;
            var darkest = a.Max() / 255f;
            var edge = Enumerable.Range(0, n).Max(i => Mathf.Max(Mathf.Max(a[i], a[(n - 1) * n + i]), Mathf.Max(a[i * n], a[i * n + n - 1]))) / 255f;
            var centre = a[n / 2 * n + n / 2] / 255f;
            expect(shadow.Quad != null && shadow.DeskHeight < lowest && lowest - shadow.DeskHeight < 0.002f,
                $"contact shadow: desk {(lowest - shadow.DeskHeight) * 1000f:0.0} mm under the heart's lowest point");
            expect(darkest > 0.6f && centre > 0.1f && edge < 0.02f,
                $"contact shadow: darkest {darkest:0.00}, under the middle {centre:0.00}, at its edges {edge:0.00}");
            shadow.Clear();                       // edit mode: OnDisable does not run
            UnityEngine.Object.DestroyImmediate(shadow);
        }

        static void CheckManipulator(GameObject inst, Action<bool, string> expect)
        {
            var shadow = inst.AddComponent<HeartContactShadow>();
            shadow.shader = Shader.Find(k_ShadowShader);
            var m = inst.AddComponent<HeartManipulator>();
            m.Init();
            var t = inst.transform;
            var home = (t.localPosition, t.localRotation, t.localScale);
            var resting = m.DeskHeight + m.gap;
            float Lowest() => HeartContactShadow.Vertices(t, t.parent).Min(p => p.y);
            Vector3 Centre() => m.TargetCentre;
            var centre0 = Centre();

            m.Orbit(90f, 0f, Vector3.right);
            m.Snap();
            var turned = Quaternion.Angle(home.Item2, t.localRotation);
            var flat = new Vector2(Centre().x - centre0.x, Centre().z - centre0.z).magnitude;
            expect(Mathf.Abs(turned - 90f) < 0.01f && flat < 1e-5f && Mathf.Abs(Lowest() - resting) < 0.0005f,
                $"rotate 90 deg about the vertical: turned {turned:0.00} deg about its centre (moved {flat * 1000f:0.000} mm), " +
                $"lowest point {(Lowest() - m.DeskHeight) * 1000f:0.00} mm above the desk");

            m.Orbit(0f, 60f, Vector3.right);
            m.Zoom(2f);
            m.Snap();
            var quad = shadow.Quad != null ? shadow.Quad.transform : null;
            var quadFlat = quad != null && Vector3.Angle(quad.up, Vector3.up) < 0.01f
                           && Mathf.Abs(quad.GetComponent<MeshFilter>().sharedMesh.bounds.center.y - m.DeskHeight) < 1e-5f;
            expect(Mathf.Abs(t.localScale.x / home.Item3.x - 2f) < 1e-4f && Mathf.Abs(Lowest() - resting) < 0.0005f && quadFlat,
                $"tilt 60 deg and zoom 2x: scale x{t.localScale.x / home.Item3.x:0.00}, still resting on the desk " +
                $"({(Lowest() - m.DeskHeight) * 1000f:0.00} mm), contact shadow rebuilt flat on the desk");

            m.Move(new Vector3(0.05f, 0.3f, 0f));
            m.Snap();
            expect(Mathf.Abs(Centre().x - centre0.x - 0.05f) < 1e-5f && Mathf.Abs(Lowest() - resting) < 0.0005f,
                "move: 5 cm across the desk; the upward part is ignored while it rests on the desk");

            m.Zoom(100f);
            var capped = m.TargetScale;
            m.ResetView();
            m.Snap();
            expect(Mathf.Approximately(capped, m.maxScale) && (t.localPosition - home.Item1).magnitude < 1e-5f
                   && Quaternion.Angle(t.localRotation, home.Item2) < 0.01f && (t.localScale - home.Item3).magnitude < 1e-6f,
                $"zoom capped at x{capped:0.0}; reset view restores the original pose");
            shadow.Clear();
            UnityEngine.Object.DestroyImmediate(m);
            UnityEngine.Object.DestroyImmediate(shadow);
        }

        static Material EnsureDeskMaterial()
        {
            var path = $"{MaterialsFolder}/Desk.mat";
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m != null) return m;
            m = new Material(Shader.Find("Standard")) { color = new Color(0.60f, 0.58f, 0.55f) };
            m.SetFloat("_Glossiness", 0.25f);
            AssetDatabase.CreateAsset(m, path);
            return m;
        }

        /// <summary>Material assets of the realistic look (Assets/Heart/Materials), created when missing.</summary>
        public static Dictionary<string, Material> EnsureMaterials()
        {
            if (!AssetDatabase.IsValidFolder(MaterialsFolder)) AssetDatabase.CreateFolder("Assets/Heart", "Materials");
            var mats = new Dictionary<string, Material>();
            foreach (var (file, shaderName, color) in k_Materials)
            {
                var path = $"{MaterialsFolder}/{file}.mat";
                var m = AssetDatabase.LoadAssetAtPath<Material>(path);
                if (m == null)
                {
                    var shader = Shader.Find(shaderName);
                    if (shader == null) throw new InvalidOperationException($"shader {shaderName} not found");
                    m = new Material(shader);
                    m.SetColor("baseColorFactor", color);
                    AssetDatabase.CreateAsset(m, path);
                }
                mats[file] = m;
            }
            AssetDatabase.SaveAssets();
            return mats;
        }

        /// <summary>HeartMaterials on the heart root, with the material assets assigned.</summary>
        public static HeartMaterials AddMaterials(GameObject heart)
        {
            var mats = EnsureMaterials();
            var look = heart.GetComponent<HeartMaterials>();
            if (look == null) look = heart.AddComponent<HeartMaterials>();
            look.ventricleMuscle = mats["Ventricle muscle"];
            look.atrialMuscle = mats["Atrial muscle"];
            look.wall = mats["Wall"];
            look.blood = mats["Blood"];
            look.oxygenated = mats["Oxygenated vessel"];
            look.deoxygenated = mats["Deoxygenated vessel"];
            look.fat = mats["Fat"];
            return look;
        }

        /// <summary>Enclosed volume (mL) of the LV mesh as currently deformed by its blend shapes.</summary>
        static float LvVolumeMl(SkinnedMeshRenderer r)
        {
            var baked = new Mesh();
            r.BakeMesh(baked, true);
            var v = baked.vertices;
            var t = baked.triangles;
            double vol = 0;
            for (var i = 0; i < t.Length; i += 3)
                vol += Vector3.Dot(v[t[i]], Vector3.Cross(v[t[i + 1]], v[t[i + 2]]));
            UnityEngine.Object.DestroyImmediate(baked);
            return (float)(Math.Abs(vol) / 6.0 * 1e6);
        }

        /// <summary>
        /// A scene with the heart (HeartController, HeartMaterials, HeartContactShadow, HeartManipulator and
        /// HeartMouseInput, HeartSlice, HeartPicker, HeartUI) on a desk, both on a stage (HeartStage: quality, comfort,
        /// case picker), soft room lighting, and a camera in front of it and a little above. Turn the desk off for
        /// mixed reality: the shadow then falls on the real table.
        /// </summary>
        public static bool CreateDemoScene()
        {
            var root = AssetDatabase.LoadAssetAtPath<GameObject>(GlbPath);
            var clip = AssetDatabase.LoadAllAssetsAtPath(GlbPath).OfType<AnimationClip>().FirstOrDefault(c => c.name == "CardiacCycle");
            if (root == null || clip == null) return false;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var stage = new GameObject("Heart Stage");                       // the desk (or table) the heart sits on
            var heart = (GameObject)PrefabUtility.InstantiatePrefab(root, scene);
            heart.name = "Heart";
            heart.transform.SetParent(stage.transform, false);
            var ctrl = heart.AddComponent<HeartController>();
            ctrl.cardiacCycle = clip;
            var look = AddMaterials(heart);
            look.Apply();
            var shadow = heart.AddComponent<HeartContactShadow>();
            shadow.shader = Shader.Find(k_ShadowShader);
            heart.AddComponent<HeartManipulator>();
            heart.AddComponent<HeartMouseInput>();
            heart.AddComponent<HeartSlice>().capShader = Shader.Find(SliceCapShader);
            heart.AddComponent<HeartPicker>();
            heart.AddComponent<HeartUI>().glassShader = Shader.Find(GlassShader);
            var hs = stage.AddComponent<HeartStage>();                       // quality, comfort, case picker
            hs.ventricleMuscle = look.ventricleMuscle; hs.atrialMuscle = look.atrialMuscle; hs.wall = look.wall;
            hs.blood = look.blood; hs.oxygenated = look.oxygenated; hs.deoxygenated = look.deoxygenated; hs.fat = look.fat;
            hs.contactShadowShader = shadow.shader;
            hs.sliceCapShader = Shader.Find(SliceCapShader);
            hs.uiGlassShader = Shader.Find(GlassShader);

            // the desk: just under the heart's lowest point, where the contact shadow falls
            var bounds = new Bounds();
            foreach (var r in heart.GetComponentsInChildren<SkinnedMeshRenderer>())
                if (bounds.size == Vector3.zero) bounds = r.sharedMesh.bounds; else bounds.Encapsulate(r.sharedMesh.bounds);
            var deskY = HeartContactShadow.LocalVertices(heart.transform).Min(p => p.y) - shadow.gap;
            var desk = GameObject.CreatePrimitive(PrimitiveType.Plane);
            desk.name = "Desk (turn off in mixed reality)";
            UnityEngine.Object.DestroyImmediate(desk.GetComponent<Collider>());
            desk.transform.SetParent(stage.transform, false);
            desk.transform.position = new Vector3(bounds.center.x, deskY, bounds.center.z);
            desk.transform.localScale = new Vector3(0.1f, 1f, 0.1f);                // 1 m x 1 m
            desk.GetComponent<MeshRenderer>().sharedMaterial = EnsureDeskMaterial();

            // glTF +Z = anterior: the camera is in front of the patient and a little above, looking at the heart,
            // far enough for the heart to fill ~70 % of the view height
            var cam = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener)).GetComponent<Camera>();
            cam.tag = "MainCamera";
            var distance = bounds.extents.y / Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad) / 0.7f;
            var view = Quaternion.Euler(k_ViewPitch, 180f, 0f);
            cam.transform.SetPositionAndRotation(bounds.center - view * Vector3.forward * (bounds.extents.z + distance), view);
            cam.nearClipPlane = 0.01f;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.20f, 0.21f, 0.23f);                     // dim room

            // soft room light: a gradient ambient (bright above, darker towards the floor), a warm key light from
            // above the front, a cool fill from the side, a rim from behind. No shadow maps: the contact shadow
            // grounds the heart, and real-time shadows are costly in a headset
            foreach (var (name, intensity, colour, rotation) in new[]
            {
                ("Key Light", 1.0f, new Color(1f, 0.96f, 0.9f), new Vector3(45f, 160f, 0f)),
                ("Fill Light", 0.35f, new Color(0.85f, 0.9f, 1f), new Vector3(10f, 240f, 0f)),
                ("Rim Light", 0.45f, Color.white, new Vector3(25f, -20f, 0f)),
            })
            {
                var l = new GameObject(name, typeof(Light)).GetComponent<Light>();
                l.type = LightType.Directional;
                l.intensity = intensity;
                l.color = colour;
                l.shadows = LightShadows.None;
                l.transform.rotation = Quaternion.Euler(rotation);
            }
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.36f, 0.36f, 0.38f);
            RenderSettings.ambientEquatorColor = new Color(0.26f, 0.25f, 0.25f);
            RenderSettings.ambientGroundColor = new Color(0.13f, 0.12f, 0.11f);
            return EditorSceneManager.SaveScene(scene, ScenePath);
        }

        [MenuItem("Tools/Heart/Render preview images")]
        public static void RenderPreviewFromMenu() => Debug.Log(RenderPreview());

        /// <summary>
        /// Command-line entry point (run without -nographics, so the GPU renders): pictures of the demo scene in
        /// Logs/heart_preview_*.png: realistic look at rest and at mid-systole (opaque wall), realistic with the
        /// see-through wall, the labelled look, a close-up, and turned and zoomed with the mouse controls.
        /// </summary>
        public static void RunRenderPreview()
        {
            string message;
            var ok = false;
            try { message = RenderPreview(); ok = true; }
            catch (Exception e) { message = "EXCEPTION " + e; }
            Debug.Log(message);
            EditorApplication.Exit(ok ? 0 : 1);
        }

        static string RenderPreview()
        {
            if (!CreateDemoScene()) throw new InvalidOperationException("no demo scene");
            EditorSceneManager.OpenScene(ScenePath);
            var ctrl = UnityEngine.Object.FindAnyObjectByType<HeartController>();
            var look = ctrl.GetComponent<HeartMaterials>();
            var shadow = ctrl.GetComponent<HeartContactShadow>();
            var manipulator = ctrl.GetComponent<HeartManipulator>();
            var ui = ctrl.GetComponent<HeartUI>();
            if (ui != null) ui.enabled = false;              // the interface has its own preview (HeartUICheck)
            if (manipulator != null) manipulator.Init();
            if (shadow != null) shadow.Build();            // edit mode: OnEnable does not run
            var cam = Camera.main;
            var fov = cam.fieldOfView;
            var rt = new RenderTexture(900, 900, 24) { antiAliasing = 4 };
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            var written = new List<string>();
            ctrl.Play();
            try
            {
                foreach (var (phase, label, realistic, opacity) in new[]
                {
                    (0f, "rest", true, 1f), (k_SystolePhase, "systole", true, 1f),
                    (0f, "see_through", true, 0.45f), (0f, "labelled", false, 0.45f), (0f, "closeup", true, 1f),
                    (0f, "turned", true, 1f),
                })
                {
                    if (label == "turned" && manipulator != null)
                    {
                        manipulator.Orbit(120f, 35f, Vector3.right);        // as a mouse drag would
                        manipulator.Zoom(1.3f);
                        manipulator.Snap();
                    }
                    cam.fieldOfView = label == "closeup" ? fov / 2.5f : fov;
                    look.CurrentLook = realistic ? HeartMaterials.Look.Realistic : HeartMaterials.Look.Labelled;
                    ctrl.wallOpacity = opacity;
                    ctrl.SetPhase(phase);
                    ctrl.Apply();
                    var baked = Bake(ctrl.gameObject);
                    cam.targetTexture = rt;
                    cam.Render();
                    Unbake(baked);
                    RenderTexture.active = rt;
                    tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                    tex.Apply();
                    var path = $"Logs/heart_preview_{label}.png";
                    File.WriteAllBytes(path, tex.EncodeToPNG());
                    written.Add(path);
                }
            }
            finally
            {
                cam.targetTexture = null;
                RenderTexture.active = null;
                ctrl.Stop();
                if (shadow != null) shadow.Clear();
                UnityEngine.Object.DestroyImmediate(rt);
                UnityEngine.Object.DestroyImmediate(tex);
            }
            return $"rendered with {SystemInfo.graphicsDeviceName}: {string.Join(", ", written)}";
        }

        /// <summary>
        /// Outside play mode Unity does not re-skin for a one-off Camera.Render, so draw a baked copy of each visible,
        /// deformed mesh instead (in play mode or in a headset the animation deforms live). Undo with Unbake.
        /// </summary>
        public static List<(SkinnedMeshRenderer smr, GameObject copy)> Bake(GameObject heart)
        {
            var baked = new List<(SkinnedMeshRenderer, GameObject)>();
            foreach (var smr in heart.GetComponentsInChildren<SkinnedMeshRenderer>())
            {
                if (!smr.enabled) continue;
                var copy = new GameObject(smr.name + " (baked)", typeof(MeshFilter), typeof(MeshRenderer));
                copy.transform.SetPositionAndRotation(smr.transform.position, smr.transform.rotation);
                var mesh = new Mesh { indexFormat = smr.sharedMesh.indexFormat };
                smr.BakeMesh(mesh, true);
                mesh.colors = smr.sharedMesh.colors;                                       // baked AO
                mesh.SetUVs(HeartMaterials.RestUvChannel, smr.sharedMesh.vertices);        // rest positions
                copy.GetComponent<MeshFilter>().sharedMesh = mesh;
                var mr = copy.GetComponent<MeshRenderer>();
                mr.sharedMaterials = smr.sharedMaterials;
                var block = new MaterialPropertyBlock();
                smr.GetPropertyBlock(block);
                mr.SetPropertyBlock(block);
                smr.enabled = false;
                baked.Add((smr, copy));
            }
            return baked;
        }

        public static void Unbake(List<(SkinnedMeshRenderer smr, GameObject copy)> baked)
        {
            foreach (var (smr, copy) in baked)
            {
                UnityEngine.Object.DestroyImmediate(copy.GetComponent<MeshFilter>().sharedMesh);
                UnityEngine.Object.DestroyImmediate(copy);
                smr.enabled = true;
            }
        }

        static string Finish(List<string> lines, int fails, out bool ok)
        {
            ok = fails == 0;
            lines.Insert(0, ok ? $"ALL {lines.Count} CHECKS PASSED" : $"{fails} OF {lines.Count} CHECKS FAILED");
            return string.Join("\n", lines);
        }
    }
}
