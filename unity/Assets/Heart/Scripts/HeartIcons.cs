// UTM-Claude Research Project | rohayanti@utm.my
// The interface's icons, drawn at runtime from simple shapes (signed distances, anti-aliased), so the project
// needs no image assets: rotate, zoom, slice, play, pause, reset, recentre, heart, caret.
using System;
using System.Collections.Generic;
using UnityEngine;

namespace UTM.Heart
{
    public static class HeartIcons
    {
        const int k_Size = 96;
        static readonly Dictionary<string, Sprite> s_Cache = new Dictionary<string, Sprite>();

        public static readonly string[] Names = { "rotate", "zoom", "slice", "play", "pause", "reset", "recentre", "heart", "caret" };

        /// <summary>A white icon (alpha = shape) for an Image; tint it with the Image's colour.</summary>
        public static Sprite Get(string name)
        {
            if (s_Cache.TryGetValue(name, out var s) && s != null) return s;
            var shape = Shape(name);
            var tex = new Texture2D(k_Size, k_Size, TextureFormat.RGBA32, false)
                { name = "Icon " + name, wrapMode = TextureWrapMode.Clamp, hideFlags = HideFlags.DontSave };
            var px = new Color32[k_Size * k_Size];
            var aa = 1.2f / k_Size;
            for (var y = 0; y < k_Size; y++)
            for (var x = 0; x < k_Size; x++)
            {
                var p = new Vector2((x + 0.5f) / k_Size - 0.5f, (y + 0.5f) / k_Size - 0.5f);
                var a = Mathf.Clamp01(0.5f - shape(p) / aa);
                px[y * k_Size + x] = new Color32(255, 255, 255, (byte)Mathf.RoundToInt(a * 255f));
            }
            tex.SetPixels32(px);
            tex.Apply(false);
            s = Sprite.Create(tex, new Rect(0, 0, k_Size, k_Size), new Vector2(0.5f, 0.5f), 100f);
            s.hideFlags = HideFlags.DontSave;
            s_Cache[name] = s;
            return s;
        }

        /// <summary>Signed distance of an icon's shape in [-0.5, 0.5]² (negative inside).</summary>
        static Func<Vector2, float> Shape(string name)
        {
            const float w = 0.045f;                                       // stroke half-width
            switch (name)
            {
                case "rotate":   // an open orbit round a small ball, ending in an arrow
                    return p => Mathf.Min(Mathf.Min(p.x > 0.12f && p.y > -0.02f ? 1f : Ellipse(p, 0.36f, 0.17f) - w * 0.85f,
                                                    Circle(p, Vector2.zero, 0.09f)),
                                          Triangle(p, new Vector2(0.06f, 0.26f), new Vector2(0.24f, 0.13f), new Vector2(0.05f, 0.04f)));
                case "zoom":
                    return p => Mathf.Min(Mathf.Abs(Circle(p, new Vector2(-0.07f, 0.07f), 0.2f)) - w,
                                          Segment(p, new Vector2(0.09f, -0.09f), new Vector2(0.3f, -0.3f)) - w * 1.5f);
                case "slice":    // a block with a plane through it
                    return p => Mathf.Min(Mathf.Abs(Box(p, new Vector2(0f, -0.06f), new Vector2(0.24f, 0.22f))) - w * 0.8f,
                                          Segment(p, new Vector2(-0.42f, 0.06f), new Vector2(0.42f, 0.2f)) - w * 1.1f);
                case "play":
                    return p => Triangle(p, new Vector2(-0.18f, -0.28f), new Vector2(-0.18f, 0.28f), new Vector2(0.3f, 0f));
                case "pause":
                    return p => Mathf.Min(Box(p, new Vector2(-0.13f, 0f), new Vector2(0.07f, 0.27f)), Box(p, new Vector2(0.13f, 0f), new Vector2(0.07f, 0.27f)));
                case "reset":    // a circular arrow
                    return p => Mathf.Min(Arc(p, 0.27f, 50f, 330f) - w,
                                          Triangle(p, new Vector2(0.06f, 0.28f), new Vector2(0.28f, 0.42f), new Vector2(0.3f, 0.12f)));
                case "recentre": // a target
                    return p => Mathf.Min(Mathf.Min(Mathf.Abs(Circle(p, Vector2.zero, 0.24f)) - w, Circle(p, Vector2.zero, 0.07f)),
                                   Mathf.Min(Mathf.Min(Segment(p, new Vector2(0f, 0.3f), new Vector2(0f, 0.44f)), Segment(p, new Vector2(0f, -0.3f), new Vector2(0f, -0.44f))),
                                             Mathf.Min(Segment(p, new Vector2(0.3f, 0f), new Vector2(0.44f, 0f)), Segment(p, new Vector2(-0.3f, 0f), new Vector2(-0.44f, 0f)))) - w);
                case "heart":
                    return p => Mathf.Min(Mathf.Min(Circle(p, new Vector2(-0.13f, 0.09f), 0.17f), Circle(p, new Vector2(0.13f, 0.09f), 0.17f)),
                                          Triangle(p, new Vector2(-0.29f, 0.03f), new Vector2(0.29f, 0.03f), new Vector2(0f, -0.34f)));
                case "caret":
                    return p => Triangle(p, new Vector2(-0.2f, 0.1f), new Vector2(0.2f, 0.1f), new Vector2(0f, -0.14f));
                default:
                    return p => Circle(p, Vector2.zero, 0.2f);
            }
        }

        static float Circle(Vector2 p, Vector2 c, float r) => (p - c).magnitude - r;

        static float Ellipse(Vector2 p, float a, float b) => Mathf.Abs(new Vector2(p.x / a, p.y / b).magnitude - 1f) * Mathf.Min(a, b);

        static float Box(Vector2 p, Vector2 c, Vector2 half)
        {
            var d = new Vector2(Mathf.Abs(p.x - c.x) - half.x, Mathf.Abs(p.y - c.y) - half.y);
            return new Vector2(Mathf.Max(d.x, 0f), Mathf.Max(d.y, 0f)).magnitude + Mathf.Min(Mathf.Max(d.x, d.y), 0f);
        }

        static float Segment(Vector2 p, Vector2 a, Vector2 b)
        {
            var ab = b - a;
            var t = Mathf.Clamp01(Vector2.Dot(p - a, ab) / ab.sqrMagnitude);
            return (p - (a + t * ab)).magnitude;
        }

        /// <summary>Distance to a circular arc of radius r from angle a0 to a1 (degrees, anticlockwise from +x).</summary>
        static float Arc(Vector2 p, float r, float a0, float a1)
        {
            var ang = Mathf.Repeat(Mathf.Atan2(p.y, p.x) * Mathf.Rad2Deg - a0, 360f);
            if (ang <= a1 - a0) return Mathf.Abs(p.magnitude - r);
            Vector2 End(float deg) => new Vector2(Mathf.Cos(deg * Mathf.Deg2Rad), Mathf.Sin(deg * Mathf.Deg2Rad)) * r;
            return Mathf.Min((p - End(a0)).magnitude, (p - End(a1)).magnitude);
        }

        static float Triangle(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
        {
            var d = Mathf.Min(Mathf.Min(Segment(p, a, b), Segment(p, b, c)), Segment(p, c, a));
            float Cross(Vector2 u, Vector2 v) => u.x * v.y - u.y * v.x;
            var s1 = Cross(b - a, p - a); var s2 = Cross(c - b, p - b); var s3 = Cross(a - c, p - c);
            var inside = (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
            return inside ? -d : d;
        }
    }
}
