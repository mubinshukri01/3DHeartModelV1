// UTM-Claude Research Project | rohayanti@utm.my
// Rotate, zoom and move the heart about its own centre, smoothly, and reset the view. It does not read any
// input itself: the mouse (HeartMouseInput) drives it now, and hand tracking can call the same methods later.
// By default the heart rests on the desk: after every change it is raised or lowered so its lowest point
// touches the desk, and the contact shadow is rebuilt for the new pose.
using UnityEngine;

namespace UTM.Heart
{
    /// <summary>
    /// Put this on the root of an imported heart_animated.glb. All poses are in the space of the heart's parent
    /// (the "stage": the desk, or a mixed-reality anchor; the scene when there is none), with +Y up from the desk.
    /// </summary>
    [DisallowMultipleComponent]
    public class HeartManipulator : MonoBehaviour
    {
        const int k_ProbePoints = 8000;              // rest vertices used to find the lowest point while moving

        [Tooltip("After every rotation, zoom or move, lower or raise the heart so it rests on the desk")]
        public bool restOnDesk = true;
        [Range(0.2f, 1f)] public float minScale = 0.5f;
        [Range(1f, 10f)] public float maxScale = 4f;
        [Tooltip("How fast the heart follows the input (1/s); 0 = at once")]
        public float smoothing = 14f;
        [Tooltip("Gap between the heart's lowest point and the desk (m)")]
        public float gap = 0.0005f;
        [Tooltip("Seconds between contact-shadow rebuilds while the heart is moving (it is also rebuilt when it stops)")]
        public float shadowInterval = 0.2f;

        Vector3 m_LocalCentre;                       // centre of the heart in its own space (rest shape)
        Vector3[] m_Probe, m_All;                    // relative to the centre: a sample, and every vertex
        bool m_Precise;                              // grounded on every vertex since the last change
        Vector3 m_HomeCentre, m_HomeScale;
        Quaternion m_HomeRotation;
        Vector3 m_Centre, m_TargetCentre;
        Quaternion m_Rotation, m_TargetRotation;
        float m_Scale = 1f, m_TargetScale = 1f;
        bool m_ShadowStale;
        float m_LastShadow = float.NegativeInfinity;
        HeartContactShadow m_Shadow;

        public bool IsReady { get; private set; }
        /// <summary>Height of the desk in the stage's space.</summary>
        public float DeskHeight { get; private set; }
        public Vector3 TargetCentre => m_TargetCentre;
        public Quaternion TargetRotation => m_TargetRotation;
        /// <summary>Zoom relative to the imported size (1 = life size).</summary>
        public float TargetScale => m_TargetScale;
        public bool Settled => Quaternion.Angle(m_Rotation, m_TargetRotation) < 0.05f
                               && Mathf.Abs(m_Scale - m_TargetScale) < 1e-4f && (m_Centre - m_TargetCentre).sqrMagnitude < 1e-10f;

        void OnEnable()
        {
            if (Application.isPlaying && !IsReady) Init();
        }

        /// <summary>Take the current pose as home and the desk just under the heart (called on enable in play mode).</summary>
        public void Init()
        {
            var points = HeartContactShadow.Vertices(transform, transform);
            if (points.Count == 0) return;
            var bounds = new Bounds(points[0], Vector3.zero);
            foreach (var p in points) bounds.Encapsulate(p);
            m_LocalCentre = bounds.center;
            var step = Mathf.Max(1, points.Count / k_ProbePoints);
            m_All = new Vector3[points.Count];
            for (var i = 0; i < m_All.Length; i++) m_All[i] = points[i] - m_LocalCentre;
            m_Probe = new Vector3[(points.Count + step - 1) / step];
            for (var i = 0; i < m_Probe.Length; i++) m_Probe[i] = m_All[i * step];
            m_HomeRotation = m_Rotation = m_TargetRotation = transform.localRotation;
            m_HomeScale = transform.localScale;
            m_Scale = m_TargetScale = 1f;
            m_HomeCentre = m_Centre = m_TargetCentre = transform.localPosition + transform.localRotation * Vector3.Scale(m_HomeScale, m_LocalCentre);
            DeskHeight = m_HomeCentre.y + LowestOffset(m_HomeRotation, 1f, m_All) - gap;
            m_Precise = true;
            IsReady = true;
        }

        // ── commands ──────────────────────────────────────────────────────────
        /// <summary>Turn about the vertical (yaw) and about a horizontal world axis, e.g. the camera's right (pitch).</summary>
        public void Orbit(float yawDegrees, float pitchDegrees, Vector3 pitchAxisWorld)
        {
            Rotate(Quaternion.AngleAxis(yawDegrees, Vector3.up) * Quaternion.AngleAxis(pitchDegrees, ToStage(pitchAxisWorld)));
        }

        /// <summary>Rotate by a rotation given in the stage's space (e.g. from two hands), about the heart's centre.</summary>
        public void Rotate(Quaternion stageRotation)
        {
            m_TargetRotation = (stageRotation * m_TargetRotation).normalized;
            Ground();
        }

        public void Zoom(float factor)
        {
            m_TargetScale = Mathf.Clamp(m_TargetScale * factor, minScale, maxScale);
            Ground();
        }

        /// <summary>Move by a world-space offset (across the desk only, while the heart rests on it).</summary>
        public void Move(Vector3 worldOffset)
        {
            var d = ToStage(worldOffset);
            if (restOnDesk) d.y = 0f;
            m_TargetCentre += d;
            Ground();
        }

        public void ResetView()
        {
            m_TargetRotation = m_HomeRotation;
            m_TargetScale = 1f;
            m_TargetCentre = m_HomeCentre;
            Ground();
        }

        /// <summary>Jump to the target pose at once (no smoothing) and rebuild the contact shadow.</summary>
        public void Snap()
        {
            Ground(true);
            m_Rotation = m_TargetRotation; m_Scale = m_TargetScale; m_Centre = m_TargetCentre;
            ApplyPose();
            RebuildShadow();
        }

        // ── pose ──────────────────────────────────────────────────────────────
        void LateUpdate()
        {
            if (!IsReady) return;
            if (Settled && !m_Precise) Ground(true);        // the sample may miss the lowest point by ~1 mm
            if (!Settled)
            {
                var t = smoothing <= 0f ? 1f : 1f - Mathf.Exp(-smoothing * Time.deltaTime);
                m_Rotation = Quaternion.Slerp(m_Rotation, m_TargetRotation, t);
                m_Scale = Mathf.Lerp(m_Scale, m_TargetScale, t);
                m_Centre = Vector3.Lerp(m_Centre, m_TargetCentre, t);
                if (Settled) { m_Rotation = m_TargetRotation; m_Scale = m_TargetScale; m_Centre = m_TargetCentre; }
                ApplyPose();
            }
            if (m_ShadowStale && (Settled || Time.unscaledTime - m_LastShadow > shadowInterval)) RebuildShadow();
        }

        void ApplyPose()
        {
            transform.localRotation = m_Rotation;
            transform.localScale = m_HomeScale * m_Scale;
            transform.localPosition = m_Centre - m_Rotation * Vector3.Scale(transform.localScale, m_LocalCentre);
        }

        void RebuildShadow()
        {
            m_LastShadow = Time.unscaledTime;
            m_ShadowStale = !Settled;
            if (m_Shadow == null) m_Shadow = GetComponent<HeartContactShadow>();
            if (m_Shadow != null && m_Shadow.enabled) m_Shadow.Build();
        }

        /// <summary>Put the target pose on the desk: from the vertex sample while moving, from every vertex once still.</summary>
        void Ground(bool precise = false)
        {
            if (!IsReady) Init();
            if (restOnDesk) m_TargetCentre.y = DeskHeight + gap - LowestOffset(m_TargetRotation, m_TargetScale, precise ? m_All : m_Probe);
            m_Precise = precise;
            m_ShadowStale = true;
        }

        /// <summary>Height of the heart's lowest point below its centre for a rotation and zoom (stage space).</summary>
        float LowestOffset(Quaternion rotation, float scale, Vector3[] points)
        {
            // only the height of each rotated point is needed: the second row of the rotation matrix
            var r = Matrix4x4.Rotate(rotation);
            var s = m_HomeScale * scale;
            float ax = r.m10 * s.x, ay = r.m11 * s.y, az = r.m12 * s.z;
            var lowest = float.PositiveInfinity;
            foreach (var p in points) lowest = Mathf.Min(lowest, ax * p.x + ay * p.y + az * p.z);
            return lowest;
        }

        Vector3 ToStage(Vector3 worldDirection) =>
            transform.parent != null ? transform.parent.InverseTransformDirection(worldDirection) : worldDirection;
    }
}
