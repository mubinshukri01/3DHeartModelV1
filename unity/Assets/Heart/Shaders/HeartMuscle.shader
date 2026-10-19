// UTM-Claude Research Project | rohayanti@utm.my
// Heart muscle, opaque (atria, right ventricle): wet, glossy, with fibre striations on a helix round the
// long axis. Detail in HeartTissue.cginc.
Shader "Heart/Muscle"
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
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }
        CGPROGRAM
        #pragma surface surf Standard vertex:HT_Vert fullforwardshadows
        #pragma target 3.5
        #define HT_MUSCLE
        #include "HeartTissue.cginc"
        ENDCG
    }
    FallBack "Diffuse"
}
