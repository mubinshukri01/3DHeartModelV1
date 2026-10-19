// UTM-Claude Research Project | rohayanti@utm.my
// The heart's contact shadow (HeartContactShadow.cs): black with the shadow texture's alpha, so it darkens a
// desk in the scene or, in mixed reality, the passthrough video of the real table. Drawn before the
// transparent tissue and pulled towards the camera a little, so it wins against a desk at the same height.
Shader "Heart/Contact Shadow"
{
    Properties
    {
        _MainTex ("Shadow (alpha)", 2D) = "black" {}
        _Color ("Colour", Color) = (0, 0, 0, 1)
        _Opacity ("Opacity", Range(0, 1)) = 0.85
    }
    SubShader
    {
        Tags { "Queue" = "Transparent-100" "RenderType" = "ContactShadow" "IgnoreProjector" = "True" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off
        Offset -1, -1
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            fixed4 _Color;
            half _Opacity;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata_base v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.texcoord.xy;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return fixed4(_Color.rgb, tex2D(_MainTex, i.uv).a * _Opacity);
            }
            ENDCG
        }
    }
}
