// UTM-Claude Research Project | rohayanti@utm.my
// Heart muscle, see-through (the LV wall, label "Myocardium"): like Heart/Muscle, with the opacity in
// baseColorFactor.a (HeartController's wall opacity). A depth-only pass first, so the wall's own folds sort
// correctly and only its nearest surface blends over the chambers inside.
Shader "Heart/Muscle (see-through)"
{
    Properties
    {
        baseColorFactor ("Colour", Color) = (0.48, 0.10, 0.09, 1)
        _Gloss ("Gloss", Range(0, 1)) = 0.78
        _BumpMm ("Bump height (mm)", Range(0, 1)) = 0.25
        _Detail ("Detail contrast", Range(0, 1)) = 0.6
        _Studio ("Studio reflection", Range(0, 2)) = 0.8
        _FibreAngle ("Fibre helix angle (deg)", Range(-90, 90)) = -50
        _FibreMm ("Fibre spacing (mm)", Range(0.3, 5)) = 1.2
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Transparent" "IgnoreProjector" = "True" }
        Pass
        {
            ZWrite On
            ColorMask 0
            CGPROGRAM
            #pragma vertex HT_DepthVert
            #pragma fragment HT_DepthFrag
            #include "UnityCG.cginc"
            #include "HeartClip.cginc"
            ENDCG
        }
        CGPROGRAM
        #pragma surface surf Standard vertex:HT_Vert alpha:fade
        #pragma target 3.5
        #define HT_MUSCLE
        #include "HeartTissue.cginc"
        ENDCG
    }
    FallBack "Transparent/Diffuse"
}
