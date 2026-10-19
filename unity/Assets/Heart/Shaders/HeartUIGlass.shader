// UTM-Claude Research Project | rohayanti@utm.my
// Frosted glass for the CardioTwin XR panels (HeartUI, desktop quality): the scene behind the panel, blurred
// and tinted by the Image's colour (alpha = how much tint), with a faint light rim at the top. The sprite's
// alpha gives the rounded corners. One grab of the screen per frame, shared by all panels. Not used in the
// headset, where grabbing the screen is too costly; there the panels are plain translucent.
Shader "Heart/UI Glass"
{
    Properties
    {
        [PerRendererData] _MainTex ("Sprite", 2D) = "white" {}
        _Blur ("Blur (pixels)", Range(0, 16)) = 7
        _StencilComp ("Stencil Comparison", Float) = 8
        _Stencil ("Stencil ID", Float) = 0
        _StencilOp ("Stencil Operation", Float) = 0
        _StencilWriteMask ("Stencil Write Mask", Float) = 255
        _StencilReadMask ("Stencil Read Mask", Float) = 255
        _ColorMask ("Color Mask", Float) = 15
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" "PreviewType" = "Plane" "CanUseSpriteAtlas" = "True" }
        Stencil
        {
            Ref [_Stencil]
            Comp [_StencilComp]
            Pass [_StencilOp]
            ReadMask [_StencilReadMask]
            WriteMask [_StencilWriteMask]
        }
        Cull Off
        Lighting Off
        ZWrite Off
        ZTest [unity_GUIZTestMode]
        Blend SrcAlpha OneMinusSrcAlpha
        ColorMask [_ColorMask]

        GrabPass { "_HeartUIGrab" }

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _HeartUIGrab;
            float4 _HeartUIGrab_TexelSize;
            float _Blur;

            struct appdata
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                fixed4 color : COLOR;
                float2 uv : TEXCOORD0;
                float4 grab : TEXCOORD1;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.color = v.color;
                o.uv = v.uv;
                o.grab = ComputeGrabScreenPos(o.pos);
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float mask = tex2D(_MainTex, i.uv).a;
                float2 uv = i.grab.xy / i.grab.w;
                float2 px = _HeartUIGrab_TexelSize.xy * _Blur;
                // two rings of taps (golden-angle spiral) around the centre: a cheap, soft blur
                float3 sum = tex2D(_HeartUIGrab, uv).rgb;
                [unroll] for (int k = 1; k <= 12; k++)
                {
                    float a = k * 2.39996;
                    float r = sqrt(k / 12.0);
                    sum += tex2D(_HeartUIGrab, uv + float2(cos(a), sin(a)) * r * px).rgb;
                }
                float3 blurred = sum / 13.0;
                float3 col = lerp(blurred, i.color.rgb, i.color.a);
                col += 0.06 * smoothstep(0.55, 1.0, i.uv.y);                   // faint light from above
                return fixed4(col, mask);
            }
            ENDCG
        }
    }
}
