// UTM-Claude Research Project | rohayanti@utm.my
// Great vessels and blood: smooth, glossy, faint mottling. The colour says the blood: red oxygenated
// (aorta, pulmonary veins, LV), blue deoxygenated (venae cavae, pulmonary artery). Detail in HeartTissue.cginc.
Shader "Heart/Vessel"
{
    Properties
    {
        baseColorFactor ("Colour", Color) = (0.66, 0.10, 0.09, 1)
        _Gloss ("Gloss", Range(0, 1)) = 0.8
        _BumpMm ("Bump height (mm)", Range(0, 1)) = 0.08
        _Detail ("Detail contrast", Range(0, 1)) = 0.3
        _Studio ("Studio reflection", Range(0, 2)) = 1
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }
        CGPROGRAM
        #pragma surface surf Standard vertex:HT_Vert fullforwardshadows
        #pragma target 3.5
        #define HT_VESSEL
        #include "HeartTissue.cginc"
        ENDCG
    }
    FallBack "Diffuse"
}
