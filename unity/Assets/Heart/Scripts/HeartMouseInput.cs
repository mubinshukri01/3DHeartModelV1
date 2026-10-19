// UTM-Claude Research Project | rohayanti@utm.my
// Desktop mouse controls for the heart (HeartManipulator): left-drag rotates (sideways: about the vertical;
// up and down: about the camera's horizontal axis), zooms or moves the slice, depending on the mode (the
// toolbar's Rotate / Zoom / Slice); the scroll wheel zooms, right- or middle-drag moves the heart across the
// desk, and R or a double-click resets the view. Presses on the UI or HeartController's panel are left to them.
// Uses the Input Manager (Project Settings > Player > Active Input Handling).
using UnityEngine;
using UnityEngine.EventSystems;

namespace UTM.Heart
{
    [RequireComponent(typeof(HeartManipulator))]
    public class HeartMouseInput : MonoBehaviour
    {
        public enum DragMode { Rotate, Zoom, Slice }

        [Tooltip("What a left-drag does (the toolbar sets it)")]
        public DragMode mode = DragMode.Rotate;
        [Tooltip("Camera the heart is seen through (default: the main camera)")]
        public Camera viewCamera;
        [Tooltip("Rotation per pixel of drag (degrees)")]
        public float degreesPerPixel = 0.3f;
        [Tooltip("Zoom per notch of the scroll wheel")]
        public float zoomPerNotch = 1.1f;
        public KeyCode resetKey = KeyCode.R;

        const float k_DoubleClick = 0.3f;            // seconds
        HeartManipulator m_Manipulator;
        HeartController m_Controller;
        HeartSlice m_Slice;
        int m_Button = -1;
        Vector3 m_Last;
        float m_LastClick = float.NegativeInfinity;

        public const string Hint = "Drag: rotate · Scroll: zoom · Right-drag: move · R or double-click: reset";

        /// <summary>True while a mouse button pressed on the heart (not on the UI) is held.</summary>
        public bool Dragging => m_Button >= 0;

        void Awake()
        {
            m_Manipulator = GetComponent<HeartManipulator>();
            m_Controller = GetComponent<HeartController>();
            m_Slice = GetComponent<HeartSlice>();
        }

        void Update()
        {
            var cam = viewCamera != null ? viewCamera : Camera.main;
            if (cam == null) return;
            var mouse = Input.mousePosition;
            var onPanel = m_Controller != null && m_Controller.PanelContains(mouse)
                          || EventSystem.current != null && EventSystem.current.IsPointerOverGameObject();

            if (m_Button < 0 && !onPanel)
                for (var b = 0; b < 3; b++)
                {
                    if (!Input.GetMouseButtonDown(b)) continue;
                    m_Button = b;
                    m_Last = mouse;
                    if (b == 0)
                    {
                        if (Time.unscaledTime - m_LastClick < k_DoubleClick) m_Manipulator.ResetView();
                        m_LastClick = Time.unscaledTime;
                    }
                    break;
                }
            if (m_Button >= 0)
            {
                if (Input.GetMouseButton(m_Button))
                {
                    var d = mouse - m_Last;
                    m_Last = mouse;
                    if (d.sqrMagnitude > 0f)
                    {
                        if (m_Button != 0) m_Manipulator.Move(PanOffset(cam, d));
                        else if (mode == DragMode.Zoom) m_Manipulator.Zoom(Mathf.Pow(zoomPerNotch, d.y / 20f));
                        else if (mode == DragMode.Slice && m_Slice != null)
                        {
                            m_Slice.active = true;
                            m_Slice.position = Mathf.Clamp(m_Slice.position + d.y * 2f / Screen.height, -1f, 1f);
                        }
                        else m_Manipulator.Orbit(-d.x * degreesPerPixel, d.y * degreesPerPixel, cam.transform.right);
                    }
                }
                else m_Button = -1;
            }

            var scroll = Input.mouseScrollDelta.y;
            if (scroll != 0f && !onPanel) m_Manipulator.Zoom(Mathf.Pow(zoomPerNotch, scroll));
            if (Input.GetKeyDown(resetKey)) m_Manipulator.ResetView();
        }

        /// <summary>World offset across the desk for a drag in pixels: the heart follows the pointer.</summary>
        Vector3 PanOffset(Camera cam, Vector3 pixels)
        {
            var distance = Vector3.Distance(cam.transform.position, transform.position);
            var metresPerPixel = 2f * distance * Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad) / Screen.height;
            var right = Vector3.ProjectOnPlane(cam.transform.right, Vector3.up).normalized;
            var away = Vector3.ProjectOnPlane(cam.transform.forward, Vector3.up).normalized;
            return (right * pixels.x + away * pixels.y) * metresPerPixel;
        }
    }
}
