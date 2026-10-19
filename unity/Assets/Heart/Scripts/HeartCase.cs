// UTM-Claude Research Project | rohayanti@utm.my
// A case package from scripts/case_package.py (output/case_package/<case id>/: case.json, heart.glb, ct.raw,
// labels.raw), e.g. copied to Assets/StreamingAssets/Cases/<case id>/. Reads the case card for the Case
// panel, and the CT and label volumes (Texture3D or CPU arrays) for the slice tool, with the transform from
// the heart model's space to voxels.
using System;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace UTM.Heart
{
    [Serializable]
    public class CaseModel
    {
        public string file, units, axes, series, rest_phase;
        public string light_file;                 // the light model for standalone headsets (may be absent)
        public int triangles, light_triangles;
        public float[] origin_lps_mm;
    }

    [Serializable]
    public class CaseMotion
    {
        public string kind, clip;
        public string[] targets;
        public float bpm, period_s, lv_edv_ml, lv_esv_ml, lv_ef_percent;
    }

    [Serializable]
    public class CaseStructure
    {
        public string name, slug, kind, method;   // kind: "segmented", "approximate" or "display layer"
        public int label;                         // its value in labels.raw
        public float[] rgba;
        public float volume_ml;
        public float[] anchor_m;                  // a point well inside it (glTF space), e.g. for a callout

        public Color Color => new Color(rgba[0], rgba[1], rgba[2], rgba[3]);
        /// <summary>The anchor in the heart model's local space in Unity (glTFast mirrors glTF's X).</summary>
        public Vector3 AnchorLocal => new Vector3(-anchor_m[0], anchor_m[1], anchor_m[2]);
    }

    [Serializable]
    public class CaseMeasurement
    {
        public string name, unit, text, status, source;   // text: what the Case panel shows
        public float value;                                 // 0 when the entry has no number (e.g. "In development")
    }

    [Serializable]
    public class CaseGrid
    {
        public int[] dims;                                  // x, y, z
        public float[] spacing_mm, origin_lps_mm, direction;
        public float[] voxel_from_gltf, voxel_from_unity;   // 4x4, row-major: model position (m) -> voxel (x, y, z)
        public string order;
    }

    [Serializable]
    public class CaseWindow
    {
        public string name;
        public float center, width;
    }

    [Serializable]
    public class CaseCt
    {
        public string file, dtype, byte_order, series;
        public int hu_offset;                               // HU = stored value + hu_offset
        public CaseWindow[] windows;
    }

    [Serializable]
    public class CaseLabels
    {
        public string file, dtype, note;
    }

    [Serializable]
    public class HeartCase
    {
        public const string Schema = "cardiotwin-case/1";

        public string schema, case_id, created, notice;
        public CaseModel model;
        public CaseMotion motion;
        public CaseStructure[] structures;
        public CaseMeasurement[] measurements;
        public CaseGrid grid;
        public CaseCt ct;
        public CaseLabels labels;

        [NonSerialized] public string folder;

        /// <summary>Assets/StreamingAssets/Cases/&lt;case id&gt; (where a build finds its cases).</summary>
        public static string StreamingFolder(string caseId) => Path.Combine(Application.streamingAssetsPath, "Cases", caseId);

        public static HeartCase Load(string folder)
        {
            var c = JsonUtility.FromJson<HeartCase>(File.ReadAllText(Path.Combine(folder, "case.json")));
            if (c == null || c.schema != Schema)
                throw new InvalidDataException($"{folder}: not a {Schema} case package");
            c.folder = folder;
            return c;
        }

        public string ModelPath => Path.Combine(folder, model.file);
        public int VoxelCount => grid.dims[0] * grid.dims[1] * grid.dims[2];
        public CaseStructure Structure(string name) => structures.FirstOrDefault(s => s.name == name);
        public CaseStructure StructureByLabel(int label) => structures.FirstOrDefault(s => s.label == label);
        public CaseMeasurement Measurement(string name) => measurements.FirstOrDefault(m => m.name == name);

        // ── volumes ───────────────────────────────────────────────────────
        /// <summary>CT values as stored (HU + 1024), x fastest. The files are little-endian, like every Unity target.</summary>
        public ushort[] ReadCt()
        {
            var bytes = ReadVolume(ct.file, 2);
            var values = new ushort[VoxelCount];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        public byte[] ReadLabels() => ReadVolume(labels.file, 1);

        public float Hu(ushort stored) => stored + ct.hu_offset;

        /// <summary>CT as an R16 Texture3D: a shader sample s gives HU = s * 65535 + hu_offset.</summary>
        public Texture3D CtTexture(ushort[] values = null)
        {
            var t = new Texture3D(grid.dims[0], grid.dims[1], grid.dims[2], GraphicsFormat.R16_UNorm, TextureCreationFlags.None)
                { name = $"{case_id} CT", wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
            t.SetPixelData(values ?? ReadCt(), 0);
            t.Apply(false);
            return t;
        }

        /// <summary>Labels as an R8 Texture3D (point filtered): a shader sample s gives label = round(s * 255).</summary>
        public Texture3D LabelTexture(byte[] values = null)
        {
            var t = new Texture3D(grid.dims[0], grid.dims[1], grid.dims[2], GraphicsFormat.R8_UNorm, TextureCreationFlags.None)
                { name = $"{case_id} labels", wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Point };
            t.SetPixelData(values ?? ReadLabels(), 0);
            t.Apply(false);
            return t;
        }

        // ── model space <-> voxels ──────────────────────────────────────────
        /// <summary>Heart model's local space in Unity (m; the root of the imported heart.glb) -> voxel indices.</summary>
        public Matrix4x4 VoxelFromLocal => RowMajor(grid.voxel_from_unity);

        public Vector3 LocalToVoxel(Vector3 local) => VoxelFromLocal.MultiplyPoint3x4(local);

        /// <summary>Texture3D coordinate (0..1) of a point in the heart model's local space.</summary>
        public Vector3 LocalToUvw(Vector3 local)
        {
            var v = LocalToVoxel(local);
            return new Vector3((v.x + 0.5f) / grid.dims[0], (v.y + 0.5f) / grid.dims[1], (v.z + 0.5f) / grid.dims[2]);
        }

        /// <summary>Index into the CT / label arrays of the voxel nearest to a point (-1 outside the volume).</summary>
        public int VoxelIndex(Vector3 local)
        {
            var v = LocalToVoxel(local);
            int x = Mathf.RoundToInt(v.x), y = Mathf.RoundToInt(v.y), z = Mathf.RoundToInt(v.z);
            if (x < 0 || y < 0 || z < 0 || x >= grid.dims[0] || y >= grid.dims[1] || z >= grid.dims[2]) return -1;
            return x + grid.dims[0] * (y + grid.dims[1] * z);
        }

        byte[] ReadVolume(string file, int bytesPerVoxel)
        {
            var bytes = File.ReadAllBytes(Path.Combine(folder, file));
            if (bytes.Length != VoxelCount * bytesPerVoxel)
                throw new InvalidDataException($"{file}: {bytes.Length} bytes, expected {VoxelCount * bytesPerVoxel}");
            return bytes;
        }

        static Matrix4x4 RowMajor(float[] m)
        {
            var r = new Matrix4x4();
            for (var i = 0; i < 4; i++) r.SetRow(i, new Vector4(m[4 * i], m[4 * i + 1], m[4 * i + 2], m[4 * i + 3]));
            return r;
        }
    }
}
