// UTM-Claude Research Project | rohayanti@utm.my
// Checks a case package (scripts/case_package.py) in Assets/StreamingAssets/Cases/<case id>/ against the
// imported heart model: the case card loads, the volumes match their grid, the package's model is the
// imported one, and the CT and labels line up with the meshes in Unity's space.
// Command line (exit code 0 = all checks passed, report in Logs/case_package_check.txt):
//   Unity.exe -batchmode -nographics -projectPath <project>
//             -executeMethod UTM.Heart.EditorTools.CasePackageCheck.Run -logFile <log> [-case DEMO-001]
// In the Editor: Tools > Heart > Run case package check.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

namespace UTM.Heart.EditorTools
{
    public static class CasePackageCheck
    {
        public const string DefaultCase = "DEMO-001";
        const string ReportPath = "Logs/case_package_check.txt";
        const float k_InsetM = 0.0015f;          // mesh vertices moved this far inwards must land in their own label
        const float k_MinInsideShare = 0.85f;    // (thin sheets: wall, fat, pulmonary veins ~0.92 in Python)
        const float k_ThinInsetM = 0.00075f;     // half as far for the coronary arteries (2-4 mm wide; 0.94 in Python)
        static readonly string[] k_Thin = { "Coronary arteries" };
        static readonly string[] k_ContrastPools = { "LV blood pool", "LA blood pool", "Aorta" };

        [MenuItem("Tools/Heart/Run case package check")]
        public static void RunFromMenu() => Debug.Log(Check(DefaultCase, out _));

        public static void Run()
        {
            var args = Environment.GetCommandLineArgs();
            var i = Array.IndexOf(args, "-case");
            var caseId = i >= 0 && i + 1 < args.Length ? args[i + 1] : DefaultCase;
            string report;
            var ok = false;
            try { report = Check(caseId, out ok); }
            catch (Exception e) { report = "EXCEPTION " + e; }
            Directory.CreateDirectory(Path.GetDirectoryName(ReportPath));
            File.WriteAllText(ReportPath, report);
            Debug.Log(report);
            EditorApplication.Exit(ok ? 0 : 1);
        }

        static string Check(string caseId, out bool ok)
        {
            var lines = new List<string>();
            var fails = 0;
            void Expect(bool condition, string what)
            {
                lines.Add((condition ? "PASS " : "FAIL ") + what);
                if (!condition) fails++;
            }

            // ── case card and files ─────────────────────────────────────────
            var folder = HeartCase.StreamingFolder(caseId);
            var c = HeartCase.Load(folder);
            Expect(c.case_id == caseId && !string.IsNullOrEmpty(c.notice), $"case.json loads: case {c.case_id}, \"{c.notice}\"");
            var fat = c.Measurement("Epicardial fat volume");
            var calcium = c.measurements.FirstOrDefault(m => m.name.StartsWith("Calcium score"));
            Expect(fat != null && calcium != null && c.measurements.All(m => !string.IsNullOrEmpty(m.text)),
                $"Case panel entries: {string.Join("; ", c.measurements.Select(m => $"{m.name} = {m.text} [{m.status}]"))}");
            ushort[] ct = null;
            byte[] labels = null;
            try { ct = c.ReadCt(); labels = c.ReadLabels(); } catch (Exception e) { Expect(false, e.Message); }
            Expect(ct != null && labels != null, $"ct.raw and labels.raw match the grid {string.Join(" x ", c.grid.dims)}");
            if (ct == null || labels == null) return Finish(lines, fails, out ok);
            var same = File.Exists(c.ModelPath) && Md5(c.ModelPath) == Md5(HeartImportCheck.GlbPath);
            Expect(same, $"the package's {c.model.file} is the imported {HeartImportCheck.GlbPath}");

            // ── textures ────────────────────────────────────────────────────
            var ctTex = c.CtTexture(ct);
            var labTex = c.LabelTexture(labels);
            Expect(ctTex.width == c.grid.dims[0] && ctTex.height == c.grid.dims[1] && ctTex.depth == c.grid.dims[2]
                   && ctTex.format == TextureFormat.R16 && labTex.format == TextureFormat.R8,
                $"Texture3D: CT {ctTex.width}x{ctTex.height}x{ctTex.depth} {ctTex.format}, labels {labTex.format}");
            UnityEngine.Object.DestroyImmediate(ctTex);
            UnityEngine.Object.DestroyImmediate(labTex);

            // ── the CT and labels line up with the meshes (Unity space) ─────────
            AssetDatabase.ImportAsset(HeartImportCheck.GlbPath);
            var root = AssetDatabase.LoadAssetAtPath<GameObject>(HeartImportCheck.GlbPath);
            var smrs = root.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            var meshNames = smrs.Select(r => r.name).OrderBy(n => n).ToArray();
            Expect(meshNames.SequenceEqual(c.structures.Select(s => s.name).OrderBy(n => n)),
                $"{c.structures.Length} structures in case.json = the model's meshes");
            foreach (var s in c.structures)
            {
                var r = smrs.FirstOrDefault(x => x.name == s.name);
                if (r == null) continue;
                var m = r.sharedMesh;
                var v = m.vertices; var n = m.normals;
                var inside = 0;
                var inset = k_Thin.Contains(s.name) ? k_ThinInsetM : k_InsetM;
                for (var i = 0; i < v.Length; i++)
                {
                    var p = root.transform.InverseTransformPoint(r.transform.TransformPoint(v[i]));
                    var d = root.transform.InverseTransformDirection(r.transform.TransformDirection(n[i]));
                    var k = c.VoxelIndex(p - d.normalized * inset);
                    if (k >= 0 && labels[k] == s.label) inside++;
                }
                var share = (float)inside / v.Length;
                var a = c.VoxelIndex(s.AnchorLocal);
                var anchorOk = a >= 0 && labels[a] == s.label;
                Expect(share >= k_MinInsideShare && anchorOk,
                    $"'{s.name}': {share:P0} of vertices {inset * 1000:0.##} mm inwards are in label {s.label}; its anchor is too " +
                    $"(CT {(a >= 0 ? c.Hu(ct[a]) : float.NaN):0} HU)");
            }
            foreach (var name in k_ContrastPools)
            {
                var s = c.Structure(name);
                var a = s != null ? c.VoxelIndex(s.AnchorLocal) : -1;
                Expect(a >= 0 && c.Hu(ct[a]) > 200, $"contrast blood in the {name}: {(a >= 0 ? c.Hu(ct[a]) : float.NaN):0} HU");
            }
            var eat = c.structures.FirstOrDefault(s => s.kind == "display layer");
            if (eat != null)
            {
                var a = c.VoxelIndex(eat.AnchorLocal);
                Expect(a >= 0 && c.Hu(ct[a]) >= -190 && c.Hu(ct[a]) <= -30, $"fat window in the fat layer: {c.Hu(ct[a]):0} HU");
            }
            return Finish(lines, fails, out ok);
        }

        static string Md5(string path)
        {
            using (var md5 = MD5.Create())
            using (var fh = File.OpenRead(path))
                return BitConverter.ToString(md5.ComputeHash(fh));
        }

        static string Finish(List<string> lines, int fails, out bool ok)
        {
            ok = fails == 0;
            var head = ok ? $"ALL {lines.Count} CASE PACKAGE CHECKS PASSED" : $"{fails} OF {lines.Count} CASE PACKAGE CHECKS FAILED";
            return head + "\n" + string.Join("\n", lines);
        }
    }
}
