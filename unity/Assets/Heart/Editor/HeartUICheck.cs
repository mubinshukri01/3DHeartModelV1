// UTM-Claude Research Project | rohayanti@utm.my
// Checks the CardioTwin XR interface (Phase 3) on the demo scene: the toolbar, the Layers and Case panels,
// the slice tool with the CT on the cut and the 2D CT view, structure labels, and the world-space mode for a
// headset. Needs the case package in Assets/StreamingAssets/Cases/DEMO-001.
// Command line (exit code 0 = all checks passed, report in Logs/heart_ui_check.txt):
//   Unity.exe -batchmode -nographics -projectPath <project> -executeMethod UTM.Heart.EditorTools.HeartUICheck.Run -logFile <log>
//   Unity.exe -batchmode -projectPath <project> -executeMethod UTM.Heart.EditorTools.HeartUICheck.RunPreview -logFile <log>
//     (pictures of the interface in Logs/heart_ui_*.png; no -nographics, so the GPU renders)
// In the Editor: Tools > Heart > Run interface check.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace UTM.Heart.EditorTools
{
    public static class HeartUICheck
    {
        const string ReportPath = "Logs/heart_ui_check.txt";
        const string k_Lv = "LV blood pool", k_Fat = "Epicardial fat (EAT)";

        [MenuItem("Tools/Heart/Run interface check")]
        public static void RunFromMenu() => Debug.Log(Check(out _));

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

        /// <summary>The demo scene's heart with every component set up by hand (edit mode runs no OnEnable / Start).</summary>
        static (HeartController ctrl, HeartUI ui) OpenDemo()
        {
            if (!HeartImportCheck.CreateDemoScene()) throw new InvalidOperationException("no demo scene");
            EditorSceneManager.OpenScene(HeartImportCheck.ScenePath);
            var ctrl = UnityEngine.Object.FindAnyObjectByType<HeartController>();
            var heart = ctrl.gameObject;
            heart.GetComponent<HeartMaterials>().Apply();
            heart.GetComponent<HeartManipulator>().Init();
            heart.GetComponent<HeartContactShadow>().Build();
            heart.GetComponent<HeartSlice>().Init();
            heart.GetComponent<HeartPicker>().Build();
            var ui = heart.GetComponent<HeartUI>();
            ui.Build();
            return (ctrl, ui);
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

            var (ctrl, ui) = OpenDemo();
            var heart = ctrl.gameObject;
            var manip = heart.GetComponent<HeartManipulator>();
            var mouse = heart.GetComponent<HeartMouseInput>();
            var slice = heart.GetComponent<HeartSlice>();
            var picker = heart.GetComponent<HeartPicker>();
            var look = heart.GetComponent<HeartMaterials>();
            var c = slice.Case;
            try
            {
                Expect(c != null, $"case package loaded for the slice tool and the Case panel ({(c != null ? c.case_id : slice.CaseError)})");
                if (c == null) return Finish(lines, fails, out ok);

                // ── toolbar ─────────────────────────────────────────────────
                var keys = new[] { "Rotate", "Zoom", "Slice", "Play", "Slower", "Faster", "Reset" };
                Expect(ui.Toolbar != null && keys.All(ui.Buttons.ContainsKey) && ui.Canvas.renderMode == RenderMode.ScreenSpaceOverlay
                       && !ctrl.showDebugPanel,
                    $"toolbar: {string.Join(", ", keys)}; screen-space interface replaces the debug panel");
                Click(ui, "Play");
                var paused = !ctrl.playing;
                Click(ui, "Play");
                Expect(paused && ctrl.playing, "Play / Pause toggles the cardiac cycle");
                var rate = ctrl.heartRate;
                Click(ui, "Faster");
                var faster = ctrl.heartRate;
                Click(ui, "Slower");
                Expect(Mathf.Approximately(faster, rate + 5f) && Mathf.Approximately(ctrl.heartRate, rate),
                    $"heart rate + / -: {rate:0} -> {faster:0} -> {ctrl.heartRate:0} bpm");
                Click(ui, "Zoom");
                var zoomMode = mouse.mode;
                Click(ui, "Rotate");
                Expect(zoomMode == HeartMouseInput.DragMode.Zoom && mouse.mode == HeartMouseInput.DragMode.Rotate,
                    "Rotate / Zoom set what a drag does");
                var home = (heart.transform.localPosition, heart.transform.localRotation);
                manip.Orbit(70f, 20f, Vector3.right);
                manip.Snap();
                Click(ui, "Reset");
                manip.Snap();
                Expect((heart.transform.localPosition - home.Item1).magnitude < 1e-5f && Quaternion.Angle(heart.transform.localRotation, home.Item2) < 0.01f,
                    "Reset view restores the pose");

                // ── Layers panel ────────────────────────────────────────────
                var fat = ctrl.Structures[k_Fat];
                ui.Toggles["Fat"].isOn = false;
                var fatOff = !fat.enabled;
                ui.Toggles["Fat"].isOn = true;
                ui.Toggles["Heart"].isOn = false;
                var heartOff = ctrl.Structures.Where(kv => kv.Key != k_Fat).All(kv => !kv.Value.enabled) && fat.enabled;
                ui.Toggles["Heart"].isOn = true;
                Expect(fatOff && fat.enabled && heartOff && ctrl.Structures.Values.All(r => r.enabled),
                    "Layers: EAT overlay and Heart show / hide their structures");
                ui.Toggles[k_Lv].isOn = false;
                var lvOff = !ctrl.Structures[k_Lv].enabled;
                ui.Toggles[k_Lv].isOn = true;
                Expect(lvOff && ctrl.Structures[k_Lv].enabled, $"Layers: one toggle per structure ({ctrl.Structures.Count})");
                ui.OpacitySlider.value = 0.3f;
                Expect(Mathf.Approximately(ctrl.wallOpacity, 0.3f), "Layers: the opacity slider sets the wall opacity");
                ui.Toggles["Realistic"].isOn = false;
                var labelled = look.CurrentLook == HeartMaterials.Look.Labelled
                               && ctrl.Structures.Values.All(r => r.sharedMaterial.shader.name.StartsWith("Heart/"));
                ui.Toggles["Realistic"].isOn = true;
                Expect(labelled && look.CurrentLook == HeartMaterials.Look.Realistic, "Layers: realistic / labelled look (both can be sliced)");
                ui.Toggles["CT"].isOn = false;
                var noCt = !slice.showCt;
                ui.Toggles["CT"].isOn = true;
                Expect(noCt && slice.showCt, "Layers: CT on the cut on / off");

                // ── Case panel ──────────────────────────────────────────────
                var text = ui.CaseText;
                var wanted = new[] { $"Case {c.case_id}", "73.4 mL", "17.5", "In development", c.notice };
                Expect(wanted.All(text.Contains), $"Case panel: {string.Join(" | ", wanted)}");

                // ── slice tool through the LV ───────────────────────────────
                var lv = c.Structure(k_Lv);
                var lvLocal = lv.AnchorLocal;
                var lvWorld = heart.transform.TransformPoint(lvLocal);
                Click(ui, "Slice");
                slice.orientation = HeartSlice.Orientation.Axial;
                slice.position = (lvLocal.y - slice.LocalBounds.center.y) / slice.LocalBounds.extents.y;
                slice.Apply();
                ui.Refresh();
                var plane = Shader.GetGlobalVector("_HeartClipPlane");
                Expect(slice.active && mouse.mode == HeartMouseInput.DragMode.Slice && ui.SlicePanel.gameObject.activeSelf && slice.Cap.activeSelf
                       && slice.IsCut(lvWorld + Vector3.up * 0.01f) && !slice.IsCut(lvWorld - Vector3.up * 0.01f)
                       && Mathf.Abs(Vector4.Dot(plane, new Vector4(lvWorld.x, lvWorld.y, lvWorld.z, 1f))) < 1e-4f,
                    "Slice: an axial cut through the LV keeps the part below; the shaders get the plane; the cut face shows");
                var m = slice.CapMaterial.GetMatrix("_VoxelFromWorld");
                var fromCap = m.MultiplyPoint3x4(lvWorld);
                var expected = c.LocalToVoxel(lvLocal);
                var ct = c.ReadCt();
                var hu = c.Hu(ct[c.VoxelIndex(lvLocal)]);
                Expect((fromCap - expected).magnitude < 0.01f && hu > 200f,
                    $"CT on the cut lines up with the anatomy: the LV maps to voxel {Round(expected)} ({(fromCap - expected).magnitude:0.0000} voxel apart), {hu:0} HU (contrast blood)");
                var view = ui.CtView.texture as Texture2D;
                var d = c.grid.dims;
                var px = view != null ? view.GetPixel(Mathf.RoundToInt(expected.x), d[1] - 1 - Mathf.RoundToInt(expected.y)) : Color.black;
                Expect(view != null && view.width == d[0] && view.height == d[1] && px.r > 0.6f,
                    $"CT view: the axial CT slice ({view?.width} x {view?.height}), bright ({px.r:0.00}) at the LV");
                slice.orientation = HeartSlice.Orientation.Coronal;
                slice.position = (lvLocal.z - slice.LocalBounds.center.z) / slice.LocalBounds.extents.z;
                slice.Apply();
                ui.Refresh();
                view = ui.CtView.texture as Texture2D;
                Expect(view != null && view.width == d[0] && view.height == d[2] && Mathf.Abs(slice.WorldPlane.GetDistanceToPoint(lvWorld)) < 1e-4f,
                    $"Coronal cut through the LV; CT view {view?.width} x {view?.height}");

                // ── labels ──────────────────────────────────────────────────
                slice.orientation = HeartSlice.Orientation.Axial;
                slice.position = (lvLocal.y - slice.LocalBounds.center.y) / slice.LocalBounds.extents.y;
                slice.Apply();
                var down = new Ray(lvWorld + Vector3.up * 0.3f, Vector3.down);
                var onFace = picker.Pick(down, out var hit) ? hit : default;
                Expect(onFace.structure == k_Lv && onFace.onSliceFace && Vector3.Distance(onFace.point, lvWorld) < 0.001f,
                    $"label from above, on the cut face: {onFace.structure} (from the CT labels)");
                ui.ShowCallout(onFace, Camera.main);
                Expect(ui.CalloutVisible && ui.CalloutText.Contains(k_Lv) && ui.CalloutText.Contains("mL"), $"callout: \"{ui.CalloutText.Replace("\n", " / ")}\"");
                Click(ui, "Slice");                                         // slice off
                var top = picker.Pick(down, out hit) ? hit : default;
                ctrl.SetVisible(top.structure ?? "", false);
                var under = picker.Pick(down, out hit) ? hit : default;
                ctrl.SetVisible(top.structure ?? "", true);
                Expect(!slice.active && top.structure != null && !top.onSliceFace && under.structure != null && under.structure != top.structure
                       && Vector3.Distance(under.point, top.point) > 0f,
                    $"label without the cut: {top.structure} on top; hidden, the ray finds {under.structure} below it");

                // ── Phase 4: look, comfort, cases ───────────────────────────
                var stage = heart.GetComponentInParent<HeartStage>();
                Expect(stage != null && ui.TitleText.Contains("CardioTwin XR") && ui.TitleText.Contains($"Case {c.case_id}")
                       && ui.NoticePill.GetComponentInChildren<UnityEngine.UI.Text>().text == HeartUI.Notice,
                    $"title bar: '{ui.TitleText}'; notice '{HeartUI.Notice}'");
                var iconKeys = new[] { "Rotate", "Zoom", "Slice", "Play", "Reset", "Recentre" };
                Expect(iconKeys.All(k => ui.Buttons.ContainsKey(k) && ui.Buttons[k].transform.Find("Icon") != null
                                         && ui.Buttons[k].transform.Find("Icon").GetComponent<UnityEngine.UI.Image>().sprite != null),
                    $"toolbar icons: {string.Join(", ", iconKeys)}");
                Expect(ui.GlassOn, "desktop: frosted-glass panels (Heart/UI Glass)");
                var cases = HeartStage.AvailableCases();
                var listed = ui.CaseList.GetComponentsInChildren<UnityEngine.UI.Button>(true).Length;
                Expect(cases.Contains(c.case_id) && listed == cases.Length, $"case picker lists the case packages: {string.Join(", ", cases)}");
                Click(ui, "Standing");
                var standing = stage.posture == HeartStage.Posture.Standing;
                Click(ui, "Seated");
                Expect(standing && stage.posture == HeartStage.Posture.Seated, "Seated / Standing");
                var camT = Camera.main.transform;
                var stageHome = (stage.transform.position, stage.transform.rotation);
                stage.quality = HeartStage.Quality.Headset;
                foreach (var (posture, drop) in new[] { (HeartStage.Posture.Seated, stage.seatedDrop), (HeartStage.Posture.Standing, stage.standingDrop) })
                {
                    stage.posture = posture;
                    stage.Recentre();
                    var deskTop = stage.transform.TransformPoint(new Vector3(manip.TargetCentre.x, manip.DeskHeight, manip.TargetCentre.z));
                    var flatDistance = Vector3.ProjectOnPlane(deskTop - camT.position, Vector3.up).magnitude;
                    var facing = Vector3.Dot(stage.transform.forward, -Vector3.ProjectOnPlane(camT.forward, Vector3.up).normalized);
                    Expect(Mathf.Abs(camT.position.y - deskTop.y - drop) < 0.001f && Mathf.Abs(flatDistance - stage.reach) < 0.001f && facing > 0.999f,
                        $"Recentre, headset, {posture.ToString().ToLowerInvariant()}: desk {(camT.position.y - deskTop.y) * 100f:0.0} cm below the eyes, " +
                        $"heart {flatDistance * 100f:0.0} cm ahead, its front facing the viewer");
                }
                stage.quality = HeartStage.Quality.Desktop;
                stage.Recentre();
                Expect((stage.transform.position - stageHome.Item1).magnitude < 1e-5f && Quaternion.Angle(stage.transform.rotation, stageHome.Item2) < 0.01f,
                    "Recentre, desktop: the stage goes back where it started");

                // ── headset mode ────────────────────────────────────────────
                ui.SetRenderMode(RenderMode.WorldSpace);
                var cam = Camera.main.transform;
                var canvas = ui.Canvas.transform;
                var flatAhead = Vector3.ProjectOnPlane(cam.forward, Vector3.up).normalized;
                Expect(ui.Canvas.renderMode == RenderMode.WorldSpace && !ui.GlassOn
                       && Mathf.Abs(Vector3.ProjectOnPlane(canvas.position - cam.position, Vector3.up).magnitude - 0.85f) < 0.001f
                       && Vector3.Angle(canvas.forward, flatAhead) < 0.5f,
                    "world-space mode (headset): plain panels 0.85 m in front of the viewer, upright, facing them");
                var startPos = canvas.position;
                cam.Rotate(Vector3.up, 15f, Space.World);
                for (var i = 0; i < 30; i++) ui.FollowStep(0.05f);
                var stayed = (canvas.position - startPos).magnitude < 1e-6f;
                cam.Rotate(Vector3.up, 45f, Space.World);
                for (var i = 0; i < 90; i++) ui.FollowStep(0.05f);
                var ahead = Vector3.ProjectOnPlane(cam.forward, Vector3.up).normalized;
                var back = Vector3.Angle(Vector3.ProjectOnPlane(canvas.position - cam.position, Vector3.up), ahead);
                cam.Rotate(Vector3.up, -60f, Space.World);
                Expect(stayed && back < 2f,
                    $"panels follow lazily: still after a 15 deg turn, back in front ({back:0.0} deg) after 60 deg");
            }
            finally
            {
                slice.active = false;
                slice.Apply();
            }
            return Finish(lines, fails, out ok);
        }

        static void Click(HeartUI ui, string key)
        {
            ui.Buttons[key].onClick.Invoke();
            ui.Refresh();
        }

        static string Round(Vector3 v) => $"({v.x:0.0}, {v.y:0.0}, {v.z:0.0})";

        // ── preview ───────────────────────────────────────────────────────────
        public static void RunPreview()
        {
            string message;
            var ok = false;
            try { message = Preview(); ok = true; }
            catch (Exception e) { message = "EXCEPTION " + e; }
            Debug.Log(message);
            EditorApplication.Exit(ok ? 0 : 1);
        }

        static string Preview()
        {
            var (ctrl, ui) = OpenDemo();
            var heart = ctrl.gameObject;
            var slice = heart.GetComponent<HeartSlice>();
            var picker = heart.GetComponent<HeartPicker>();
            var cam = Camera.main;
            ui.uiCamera = cam;
            ui.SetRenderMode(RenderMode.ScreenSpaceCamera);
            var rt = new RenderTexture(1600, 900, 24) { antiAliasing = 4 };
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            var c = slice.Case;
            var lvLocal = c.Structure(k_Lv).AnchorLocal;
            var written = new List<string>();
            try
            {
                foreach (var label in new[] { "overview", "axial_cut", "coronal_cut" })
                {
                    slice.active = label != "overview";
                    slice.orientation = label == "coronal_cut" ? HeartSlice.Orientation.Coronal : HeartSlice.Orientation.Axial;
                    slice.position = label == "coronal_cut"
                        ? (lvLocal.z - slice.LocalBounds.center.z) / slice.LocalBounds.extents.z
                        : (lvLocal.y - slice.LocalBounds.center.y) / slice.LocalBounds.extents.y;
                    slice.Apply();
                    ui.Refresh();
                    // label what the pointer would be on: the LV on the cut face, or the fat in the overview
                    cam.targetTexture = rt;
                    var target = slice.active ? heart.transform.TransformPoint(lvLocal)
                                              : heart.transform.TransformPoint(c.Structure(k_Fat).AnchorLocal);
                    var ray = new Ray(cam.transform.position, (target - cam.transform.position).normalized);
                    if (picker.Pick(ray, out var hit)) ui.ShowCallout(hit, cam); else ui.ShowCallout(false);
                    Canvas.ForceUpdateCanvases();
                    var baked = HeartImportCheck.Bake(heart);
                    cam.Render();
                    HeartImportCheck.Unbake(baked);
                    RenderTexture.active = rt;
                    tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                    tex.Apply();
                    var path = $"Logs/heart_ui_{label}.png";
                    File.WriteAllBytes(path, tex.EncodeToPNG());
                    written.Add(path);
                }
            }
            finally
            {
                slice.active = false;
                slice.Apply();
                cam.targetTexture = null;
                RenderTexture.active = null;
                UnityEngine.Object.DestroyImmediate(rt);
                UnityEngine.Object.DestroyImmediate(tex);
            }
            return $"rendered with {SystemInfo.graphicsDeviceName}: {string.Join(", ", written)}";
        }

        static string Finish(List<string> lines, int fails, out bool ok)
        {
            ok = fails == 0;
            lines.Insert(0, ok ? $"ALL {lines.Count} INTERFACE CHECKS PASSED" : $"{fails} OF {lines.Count} INTERFACE CHECKS FAILED");
            return string.Join("\n", lines);
        }
    }
}
