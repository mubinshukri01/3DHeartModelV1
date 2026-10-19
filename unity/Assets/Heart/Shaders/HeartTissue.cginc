// UTM-Claude Research Project | rohayanti@utm.my
// Procedural tissue for the heart meshes (built-in render pipeline, surface shaders). Marching-cubes meshes
// have no texture coordinates, so all detail is 3D noise evaluated at each vertex's rest position
// (HeartMaterials copies it into TEXCOORD3; without it the deformed position is used and the pattern
// slides a little as the heart beats). Bumps use screen-space derivatives (Mikkelsen 2010), so no
// tangents are needed either. Shaders define one of HT_MUSCLE, HT_VESSEL or HT_FAT before including this.
#ifndef HEART_TISSUE_INCLUDED
#define HEART_TISSUE_INCLUDED

#include "HeartClip.cginc"

fixed4 baseColorFactor;     // same name as glTFast's materials, so HeartController's opacity works unchanged
half _Gloss;                // smoothness
half _BumpMm;               // bump height (mm)
half _Detail;               // contrast of the colour detail
half _Studio;               // strength of the built-in studio reflection
half _FibreAngle;           // muscle: fibre helix angle (degrees)
half _FibreMm;              // muscle: spacing of the fibre striations (mm)
half _LobuleMm;             // fat: lobule size (mm)

struct Input
{
    float3 restMm;          // rest position, object space, mm (origin = heart centre, +Y = superior)
    float4 color : COLOR;   // baked ambient occlusion (build_heart_model.py)
    float3 worldPos;
    float3 worldNormal;
    INTERNAL_DATA
};

void HT_Vert(inout appdata_full v, out Input o)
{
    UNITY_INITIALIZE_OUTPUT(Input, o);
    float3 rest = any(v.texcoord3.xyz) ? v.texcoord3.xyz : v.vertex.xyz;
    o.restMm = rest * 1000.0;
    // any tangent frame will do (detail is 3D and bumps are screen-space); surface shaders need one
    float3 t = cross(v.normal, abs(v.normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0));
    v.tangent = float4(normalize(t), 1.0);
}

// ── noise ─────────────────────────────────────────────────────────────────
float3 HT_Hash3(float3 p)       // Dave Hoskins' hash without sine: -1..1
{
    p = frac(p * float3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return frac((p.xxy + p.yxx) * p.zyx) * 2.0 - 1.0;
}

float HT_Noise(float3 p)        // gradient noise, about -0.7..0.7
{
    float3 i = floor(p), f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    float n000 = dot(HT_Hash3(i), f);
    float n100 = dot(HT_Hash3(i + float3(1, 0, 0)), f - float3(1, 0, 0));
    float n010 = dot(HT_Hash3(i + float3(0, 1, 0)), f - float3(0, 1, 0));
    float n110 = dot(HT_Hash3(i + float3(1, 1, 0)), f - float3(1, 1, 0));
    float n001 = dot(HT_Hash3(i + float3(0, 0, 1)), f - float3(0, 0, 1));
    float n101 = dot(HT_Hash3(i + float3(1, 0, 1)), f - float3(1, 0, 1));
    float n011 = dot(HT_Hash3(i + float3(0, 1, 1)), f - float3(0, 1, 1));
    float n111 = dot(HT_Hash3(i + float3(1, 1, 1)), f - float3(1, 1, 1));
    return lerp(lerp(lerp(n000, n100, u.x), lerp(n010, n110, u.x), u.y),
                lerp(lerp(n001, n101, u.x), lerp(n011, n111, u.x), u.y), u.z);
}

float HT_Fbm(float3 p)
{
    float s = 0.0, a = 0.5;
    [unroll] for (int o = 0; o < 4; o++) { s += a * HT_Noise(p); p = p * 2.03 + 17.1; a *= 0.5; }
    return s;
}

float2 HT_Voronoi(float3 p)     // distances to the nearest and second-nearest jittered cell centre (cell = 1)
{
    float3 i = floor(p), f = frac(p);
    float d1 = 8.0, d2 = 8.0;
    [unroll] for (int z = -1; z <= 1; z++)
    [unroll] for (int y = -1; y <= 1; y++)
    [unroll] for (int x = -1; x <= 1; x++)
    {
        float3 g = float3(x, y, z);
        float3 o = g + 0.5 + 0.4 * HT_Hash3(i + g) - f;
        float d = dot(o, o);
        d2 = d < d1 ? d1 : min(d2, d);
        d1 = min(d1, d);
    }
    return sqrt(float2(d1, d2));
}

// ── shading helpers ───────────────────────────────────────────────────────
float3 HT_PerturbNormal(float3 pos, float3 n, float height)    // height in world units (m)
{
    float3 dpdx = ddx(pos), dpdy = ddy(pos);
    float3 r1 = cross(dpdy, n), r2 = cross(n, dpdx);
    float det = dot(dpdx, r1);
    float3 grad = sign(det) * (ddx(height) * r1 + ddy(height) * r2);
    return normalize(abs(det) * n - grad);
}

float3 HT_Studio(float3 n, float3 v, float gloss)  // soft sky and a soft box above front-left; mixed-reality
{                                                     // rooms give no reflection probe, so the shader brings one
    float3 r = reflect(-v, n);
    float sky = smoothstep(-0.3, 0.9, r.y);
    float box = pow(saturate(dot(r, normalize(float3(-0.4, 0.6, 0.7)))), 16.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - saturate(dot(n, v)), 5.0);
    return (0.06 + 0.3 * sky + 1.2 * box) * fresnel * gloss * gloss;
}

// ── tissues ───────────────────────────────────────────────────────────────
float HT_FibreBand(float along, float y, float r, float ca, float sa, float3 seed)
{
    float s = along * ca + y * sa, n = -along * sa + y * ca;        // along / across the fibre (mm)
    float k = 1.0 / max(_FibreMm, 0.1);
    return 0.7 * HT_Noise(float3(s * 0.08, n * k, r * 0.4) + seed) + 0.3 * HT_Noise(float3(s * 0.16, n * 2.0 * k, r * 0.8) + seed);
}

float HT_Fibres(float3 p)       // striations wrapping round the long (Y) axis on a helix; two samples hide the
{                               // angle seam (each fades out at its own seam)
    float r = length(p.xz) + 1e-3, th = atan2(p.z, p.x);
    float a = radians(_FibreAngle), ca = cos(a), sa = sin(a);
    float f1 = HT_FibreBand(th * 30.0, p.y, r, ca, sa, 0.0);
    float f2 = HT_FibreBand((th < 0.0 ? th + UNITY_PI : th - UNITY_PI) * 30.0, p.y, r, ca, sa, 41.7);
    return lerp(f2, f1, 0.5 + 0.5 * cos(th));
}

void surf(Input IN, inout SurfaceOutputStandard o)
{
    HT_Clip(IN.worldPos);                                            // the slice tool's cut
    float3 p = IN.restMm;
    float3 base = baseColorFactor.rgb;
    float ao = IN.color.r;
    float detail, height, gloss = _Gloss, glow = 0.0, alpha = baseColorFactor.a;
#if defined(HT_MUSCLE)
    float fib = HT_Fibres(p), mottle = HT_Fbm(p * 0.07);
    detail = 0.25 * fib + 0.2 * mottle;
    height = 0.6 * fib + 0.4 * mottle;
    gloss *= 1.0 + 0.15 * mottle;
    glow = 0.12;
#elif defined(HT_FAT)
    float2 v = HT_Voronoi(p / max(_LobuleMm, 0.5) + 5.3);
    float dome = saturate(1.0 - 1.4 * v.x * v.x);                    // rounded lobule, highest at its centre
    float groove = 1.0 - smoothstep(0.0, 0.25, v.y - v.x);           // 1 on the border between lobules
    float grain = HT_Noise(p * 0.8);
    detail = 0.12 * dome - 0.18 * groove + 0.06 * grain;
    height = dome + 0.15 * grain;
    gloss *= lerp(0.75, 1.0, dome);
    glow = 0.12 + 0.1 * dome;                                        // light scattered inside the fat
    alpha = saturate(alpha * lerp(1.0, 1.15, groove));
#else   // HT_VESSEL
    float mottle = HT_Fbm(p * 0.1);
    detail = 0.3 * mottle;
    height = mottle;
    glow = 0.08;
#endif
    float3 nW = WorldNormalVector(IN, float3(0, 0, 1));
    float3 tW = WorldNormalVector(IN, float3(1, 0, 0));
    float3 bW = WorldNormalVector(IN, float3(0, 1, 0));
    float3 bumped = HT_PerturbNormal(IN.worldPos, nW, height * _BumpMm * 0.001);
    float3 vW = normalize(_WorldSpaceCameraPos - IN.worldPos);
    float rim = pow(1.0 - saturate(dot(bumped, vW)), 3.0);
#if defined(HT_FAT)
    alpha = saturate(alpha + 0.25 * rim);
#endif
    float3 albedo = base * (1.0 + _Detail * detail * 2.0) * lerp(1.0, ao, 0.65);
    o.Albedo = saturate(albedo);
    o.Normal = normalize(float3(dot(bumped, tW), dot(bumped, bW), dot(bumped, nW)));
    o.Smoothness = saturate(gloss);
    o.Metallic = 0.0;
    o.Occlusion = ao;
    o.Emission = base * (glow + 0.15 * rim) * ao + HT_Studio(bumped, vW, saturate(gloss)) * _Studio * ao;
    o.Alpha = alpha;
}

#endif
