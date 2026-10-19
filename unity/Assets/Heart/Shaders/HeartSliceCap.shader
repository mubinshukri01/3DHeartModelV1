// UTM-Claude Research Project | rohayanti@utm.my
// The face of the slice tool's cut (HeartSlice.cs): where the plane crosses the heart, the patient's CT
// (from the case package, on the same grid as the model), or the structures' colours from the labels. The
// heart's outline comes from a smoothed inside mask, so the face meets the cut surfaces.
Shader "Heart/Slice Cap"
{
    Properties
    {
        _CtTex ("CT (R16: HU + 1024)", 3D) = "" {}
        _LabelTex ("Labels (R8)", 3D) = "" {}
        _MaskTex ("Inside the heart (R8)", 3D) = "" {}
        _Palette ("Label colours (256 x 1)", 2D) = "black" {}
        _Window ("Window centre, width (HU)", Vector) = (300, 800, 0, 0)
        _ShowCt ("Show CT", Float) = 1
        _Tint ("Structure colour over the CT", Range(0, 1)) = 0.3
    }
    SubShader
    {
        Tags { "Queue" = "Geometry+10" "RenderType" = "Opaque" }
        Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.5
            #include "UnityCG.cginc"

            sampler3D _CtTex, _LabelTex, _MaskTex;
            sampler2D _Palette;
            float4x4 _VoxelFromWorld;   // world -> voxel index (x, y, z), set by HeartSlice
            float4 _Dims;               // voxels along x, y, z
            float4 _Window;
            float _ShowCt, _Tint, _HuOffset;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 world : TEXCOORD0;
            };

            v2f vert(appdata_base v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.world = mul(unity_ObjectToWorld, v.vertex).xyz;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float3 uvw = (mul(_VoxelFromWorld, float4(i.world, 1)).xyz + 0.5) / _Dims.xyz;
                if (any(uvw < 0) || any(uvw > 1)) discard;
                if (tex3D(_MaskTex, uvw).r < 0.5) discard;                    // outside the heart
                float label = round(tex3D(_LabelTex, uvw).r * 255.0);
                float3 tissue = tex2D(_Palette, float2((label + 0.5) / 256.0, 0.5)).rgb;
                float hu = tex3D(_CtTex, uvw).r * 65535.0 + _HuOffset;
                float g = saturate((hu - (_Window.x - 0.5 * _Window.y)) / _Window.y);
                float3 col = _ShowCt > 0.5 ? lerp(g.xxx, tissue * (0.35 + 0.65 * g), _Tint) : tissue;
                return fixed4(col, 1);
            }
            ENDCG
        }
    }
}
