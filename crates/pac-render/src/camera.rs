use glam::{Mat4, Vec3};

/// Perspective camera with FPS-style yaw/pitch controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    /// Yaw angle in radians (rotation around the Y axis).
    pub yaw: f32,
    /// Pitch angle in radians (rotation around the X axis), clamped to ±89°.
    pub pitch: f32,
    /// Vertical field-of-view in radians.
    pub fov_y: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: std::f32::consts::FRAC_PI_4, // 45°
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

/// Maximum pitch magnitude (89° in radians) to prevent gimbal-lock at the poles.
const MAX_PITCH: f32 = 89.0 * (std::f32::consts::PI / 180.0);

impl Camera {
    /// Create a camera at `position` looking along `yaw`/`pitch`.
    #[inline]
    pub fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        Self {
            position,
            yaw,
            pitch: pitch.clamp(-MAX_PITCH, MAX_PITCH),
            ..Self::default()
        }
    }

    /// Unit vector pointing in the direction the camera is looking.
    #[inline]
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    /// Unit vector pointing to the camera's right (cross of forward and world-up).
    #[inline]
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    /// Unit vector pointing up relative to the camera.
    #[inline]
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }

    /// Build the view matrix (world → camera space).
    #[inline]
    pub fn view_matrix(&self) -> Mat4 {
        let target = self.position + self.forward();
        Mat4::look_at_rh(self.position, target, Vec3::Y)
    }

    /// Build the perspective projection matrix (camera → clip space, right-handed).
    #[inline]
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Combined view-projection matrix.
    #[inline]
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Rotate the camera by yaw/pitch deltas (e.g. from mouse movement).
    /// Pitch is clamped to ±89° to prevent flipping.
    #[inline]
    pub fn rotate(&mut self, yaw_delta: f32, pitch_delta: f32) {
        self.yaw += yaw_delta;
        self.pitch = (self.pitch + pitch_delta).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Translate the camera along its local axes (FPS-style movement on the XZ plane).
    ///
    /// `forward` moves along the camera's facing direction projected onto XZ,
    /// `right` strafes, and `up` moves along world Y.
    #[inline]
    pub fn translate(&mut self, forward: f32, right: f32, up: f32) {
        // Movement forward/back is on the XZ plane (ignore pitch for FPS feel).
        let flat_forward = Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize();
        let flat_right = flat_forward.cross(Vec3::Y).normalize();

        self.position += flat_forward * forward;
        self.position += flat_right * right;
        self.position += Vec3::Y * up;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn approx_eq_vec3(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    fn approx_eq_mat4(a: Mat4, b: Mat4) -> bool {
        let ca = a.to_cols_array();
        let cb = b.to_cols_array();
        ca.iter().zip(cb.iter()).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    // ── Defaults ──────────────────────────────────────────────

    #[test]
    fn default_position_is_origin() {
        let cam = Camera::default();
        assert_eq!(cam.position, Vec3::ZERO);
    }

    #[test]
    fn default_yaw_pitch_are_zero() {
        let cam = Camera::default();
        assert_eq!(cam.yaw, 0.0);
        assert_eq!(cam.pitch, 0.0);
    }

    #[test]
    fn default_fov_is_45_degrees() {
        let cam = Camera::default();
        assert!(approx_eq(cam.fov_y, FRAC_PI_4));
    }

    // ── Constructor ───────────────────────────────────────────

    #[test]
    fn new_stores_fields() {
        let cam = Camera::new(Vec3::new(1.0, 2.0, 3.0), 0.5, 0.3);
        assert_eq!(cam.position, Vec3::new(1.0, 2.0, 3.0));
        assert!(approx_eq(cam.yaw, 0.5));
        assert!(approx_eq(cam.pitch, 0.3));
    }

    #[test]
    fn new_clamps_extreme_pitch() {
        let cam = Camera::new(Vec3::ZERO, 0.0, PI);
        assert!(cam.pitch <= MAX_PITCH);
        assert!(cam.pitch >= -MAX_PITCH);

        let cam2 = Camera::new(Vec3::ZERO, 0.0, -PI);
        assert!(cam2.pitch >= -MAX_PITCH);
    }

    // ── Direction vectors ─────────────────────────────────────

    #[test]
    fn forward_at_zero_yaw_pitch_is_positive_x() {
        let cam = Camera::default();
        let f = cam.forward();
        assert!(approx_eq_vec3(f, Vec3::X));
    }

    #[test]
    fn forward_at_90_yaw_is_positive_z() {
        let cam = Camera::new(Vec3::ZERO, FRAC_PI_2, 0.0);
        let f = cam.forward();
        assert!(approx_eq_vec3(f, Vec3::Z));
    }

    #[test]
    fn forward_is_unit_length() {
        let cam = Camera::new(Vec3::ZERO, 1.2, 0.4);
        assert!(approx_eq(cam.forward().length(), 1.0));
    }

    #[test]
    fn right_is_perpendicular_to_forward() {
        let cam = Camera::new(Vec3::ZERO, 0.7, 0.2);
        let dot = cam.forward().dot(cam.right());
        assert!(approx_eq(dot, 0.0));
    }

    #[test]
    fn up_is_perpendicular_to_forward_and_right() {
        let cam = Camera::new(Vec3::ZERO, 0.7, 0.2);
        assert!(approx_eq(cam.up().dot(cam.forward()), 0.0));
        assert!(approx_eq(cam.up().dot(cam.right()), 0.0));
    }

    // ── View matrix ───────────────────────────────────────────

    #[test]
    fn view_matrix_at_origin_default() {
        let cam = Camera::default();
        let v = cam.view_matrix();
        let expected = Mat4::look_at_rh(Vec3::ZERO, Vec3::X, Vec3::Y);
        assert!(approx_eq_mat4(v, expected));
    }

    #[test]
    fn view_matrix_changes_with_position() {
        let a = Camera::default().view_matrix();
        let b = Camera::new(Vec3::new(5.0, 0.0, 0.0), 0.0, 0.0).view_matrix();
        assert!(!approx_eq_mat4(a, b));
    }

    #[test]
    fn view_matrix_changes_with_yaw() {
        let a = Camera::new(Vec3::ZERO, 0.0, 0.0).view_matrix();
        let b = Camera::new(Vec3::ZERO, 1.0, 0.0).view_matrix();
        assert!(!approx_eq_mat4(a, b));
    }

    // ── Projection matrix ─────────────────────────────────────

    #[test]
    fn projection_matches_glam_perspective_rh() {
        let cam = Camera::default();
        let p = cam.projection_matrix();
        let expected = Mat4::perspective_rh(cam.fov_y, cam.aspect, cam.near, cam.far);
        assert!(approx_eq_mat4(p, expected));
    }

    #[test]
    fn projection_changes_with_fov() {
        let mut cam = Camera::default();
        let a = cam.projection_matrix();
        cam.fov_y = FRAC_PI_2;
        let b = cam.projection_matrix();
        assert!(!approx_eq_mat4(a, b));
    }

    #[test]
    fn projection_changes_with_aspect() {
        let mut cam = Camera::default();
        let a = cam.projection_matrix();
        cam.aspect = 4.0 / 3.0;
        let b = cam.projection_matrix();
        assert!(!approx_eq_mat4(a, b));
    }

    // ── View-projection ───────────────────────────────────────

    #[test]
    fn view_projection_is_proj_times_view() {
        let cam = Camera::new(Vec3::new(1.0, 2.0, 3.0), 0.5, 0.3);
        let vp = cam.view_projection_matrix();
        let expected = cam.projection_matrix() * cam.view_matrix();
        assert!(approx_eq_mat4(vp, expected));
    }

    // ── Rotate ────────────────────────────────────────────────

    #[test]
    fn rotate_accumulates_yaw() {
        let mut cam = Camera::default();
        cam.rotate(0.5, 0.0);
        cam.rotate(0.3, 0.0);
        assert!(approx_eq(cam.yaw, 0.8));
    }

    #[test]
    fn rotate_clamps_pitch() {
        let mut cam = Camera::default();
        cam.rotate(0.0, PI); // way past 89°
        assert!(cam.pitch <= MAX_PITCH + 1e-6);

        cam.rotate(0.0, -2.0 * PI); // way past -89°
        assert!(cam.pitch >= -MAX_PITCH - 1e-6);
    }

    // ── Translate ─────────────────────────────────────────────

    #[test]
    fn translate_forward_moves_along_facing() {
        let mut cam = Camera::default(); // facing +X
        cam.translate(1.0, 0.0, 0.0);
        assert!(approx_eq_vec3(cam.position, Vec3::X));
    }

    #[test]
    fn translate_right_strafes() {
        let mut cam = Camera::default(); // facing +X, right is +Z via cross(+X, +Y)
        cam.translate(0.0, 1.0, 0.0);
        assert!(approx_eq_vec3(cam.position, Vec3::Z));
    }

    #[test]
    fn translate_up_moves_along_world_y() {
        let mut cam = Camera::default();
        cam.translate(0.0, 0.0, 1.0);
        assert!(approx_eq_vec3(cam.position, Vec3::Y));
    }

    #[test]
    fn translate_ignores_pitch_for_forward() {
        let mut cam = Camera::new(Vec3::ZERO, 0.0, 0.5); // pitched up, facing +X
        cam.translate(1.0, 0.0, 0.0);
        // Should move along +X on the XZ plane, not upward
        assert!(approx_eq(cam.position.y, 0.0));
        assert!(approx_eq(cam.position.x, 1.0));
    }
}
