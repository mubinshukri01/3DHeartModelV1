// UTM-Claude Research Project | rohayanti@utm.my
// The slice tool's cut (HeartSlice.cs): a world-space plane set as a global shader vector. Points with
// dot(n, p) + d < 0 are cut away; the default (0, 0, 0, 0) cuts nothing. Also the depth-only pass that the
// see-through tissue draws first, cut the same way.
#ifndef HEART_CLIP_INCLUDED
#define HEART_CLIP_INCLUDED

float4 _HeartClipPlane;

inline void HT_Clip(float3 worldPos)
{
    clip(dot(_HeartClipPlane.xyz, worldPos) + _HeartClipPlane.w);
}

struct ht_depth_v2f
{
    float4 pos : SV_POSITION;
    float3 world : TEXCOORD0;
};

ht_depth_v2f HT_DepthVert(appdata_base v)
{
    ht_depth_v2f o;
    o.pos = UnityObjectToClipPos(v.vertex);
    o.world = mul(unity_ObjectToWorld, v.vertex).xyz;
    return o;
}

fixed4 HT_DepthFrag(ht_depth_v2f i) : SV_Target
{
    HT_Clip(i.world);
    return 0;
}

#endif
