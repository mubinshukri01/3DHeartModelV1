// UTM-Claude Research Project | rohayanti@utm.my
// Epicardial fat: translucent yellow lobules (cellular noise), glossy domes, darker grooves, more opaque at
// grazing angles. Drawn after the see-through wall (render queue 3001, set by HeartMaterials).
Shader "Heart/Fat"
{
    Properties
    {
        baseColorFactor ("Colour", Color) = (0.96, 0.78, 0.38, 0.72)
        _Gloss ("Gloss", Range(0, 1)) = 0.7
        _BumpMm ("Bump height (mm)", Range(0, 2)) = 0.35
        _Detail ("Detail contrast", Range(0, 1)) = 0.6
        _Studio ("Studio reflection", Range(0, 2)) = 1
        _LobuleMm ("Lobule size (mm)", Range(1, 10)) = 5
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Transparent+1" "IgnoreProjector" = "True" }
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
        #define HT_FAT
        #include "HeartTissue.cginc"
        ENDCG
    }
    FallBack "Transparent/Diffuse"
}
