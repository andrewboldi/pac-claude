use glam::{Mat4, Quat, Vec3};

/// 3D transform with position, rotation, and scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Create a transform from position, rotation, and scale.
    #[inline]
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Create a transform with only a position (identity rotation, unit scale).
    #[inline]
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            ..Self::IDENTITY
        }
    }

    /// Create a transform with only a rotation (zero position, unit scale).
    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    /// Create a transform with only a scale (zero position, identity rotation).
    #[inline]
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    /// Build the 4×4 model matrix: `T * R * S`.
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn approx_eq_mat4(a: Mat4, b: Mat4) -> bool {
        let cols_a = a.to_cols_array();
        let cols_b = b.to_cols_array();
        cols_a
            .iter()
            .zip(cols_b.iter())
            .all(|(x, y)| (x - y).abs() < 1e-6)
    }

    #[test]
    fn identity_produces_identity_matrix() {
        let t = Transform::IDENTITY;
        assert_eq!(t.to_matrix(), Mat4::IDENTITY);
    }

    #[test]
    fn default_is_identity() {
        assert_eq!(Transform::default(), Transform::IDENTITY);
    }

    #[test]
    fn from_position_sets_translation() {
        let t = Transform::from_position(Vec3::new(1.0, 2.0, 3.0));
        let m = t.to_matrix();
        assert_eq!(m, Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn from_scale_sets_scale() {
        let t = Transform::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let m = t.to_matrix();
        assert_eq!(m, Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0)));
    }

    #[test]
    fn from_rotation_sets_rotation() {
        let q = Quat::from_rotation_z(FRAC_PI_2);
        let t = Transform::from_rotation(q);
        let m = t.to_matrix();
        assert!(approx_eq_mat4(m, Mat4::from_quat(q)));
    }

    #[test]
    fn to_matrix_is_translation_times_rotation_times_scale() {
        let pos = Vec3::new(10.0, 20.0, 30.0);
        let rot = Quat::from_rotation_y(FRAC_PI_2);
        let scl = Vec3::new(2.0, 2.0, 2.0);
        let t = Transform::new(pos, rot, scl);

        let expected =
            Mat4::from_translation(pos) * Mat4::from_quat(rot) * Mat4::from_scale(scl);
        assert!(approx_eq_mat4(t.to_matrix(), expected));
    }

    #[test]
    fn new_stores_fields() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let rot = Quat::from_rotation_x(1.0);
        let scl = Vec3::new(4.0, 5.0, 6.0);
        let t = Transform::new(pos, rot, scl);
        assert_eq!(t.position, pos);
        assert_eq!(t.rotation, rot);
        assert_eq!(t.scale, scl);
    }

    #[test]
    fn non_uniform_scale() {
        let t = Transform::from_scale(Vec3::new(1.0, 2.0, 3.0));
        let m = t.to_matrix();
        // Transformed basis vectors should have the correct lengths
        let point_x = m.transform_point3(Vec3::X);
        let point_y = m.transform_point3(Vec3::Y);
        let point_z = m.transform_point3(Vec3::Z);
        assert!((point_x.length() - 1.0).abs() < 1e-6);
        assert!((point_y.length() - 2.0).abs() < 1e-6);
        assert!((point_z.length() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn from_position_leaves_rotation_and_scale_default() {
        let t = Transform::from_position(Vec3::new(5.0, 6.0, 7.0));
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn from_rotation_leaves_position_and_scale_default() {
        let t = Transform::from_rotation(Quat::from_rotation_x(1.0));
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn from_scale_leaves_position_and_rotation_default() {
        let t = Transform::from_scale(Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
    }

    #[test]
    fn to_matrix_translates_point() {
        let t = Transform::from_position(Vec3::new(10.0, 20.0, 30.0));
        let m = t.to_matrix();
        let result = m.transform_point3(Vec3::ZERO);
        assert!((result - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-6);
    }

    #[test]
    fn to_matrix_scales_point() {
        let t = Transform::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let m = t.to_matrix();
        let result = m.transform_point3(Vec3::ONE);
        assert!((result - Vec3::new(2.0, 3.0, 4.0)).length() < 1e-6);
    }

    #[test]
    fn to_matrix_rotates_point_90_deg_z() {
        let t = Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2));
        let m = t.to_matrix();
        let result = m.transform_point3(Vec3::X);
        // X rotated 90° around Z => Y
        assert!((result - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn to_matrix_rotates_point_90_deg_x() {
        let t = Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2));
        let m = t.to_matrix();
        let result = m.transform_point3(Vec3::Y);
        // Y rotated 90° around X => Z
        assert!((result - Vec3::Z).length() < 1e-5);
    }

    #[test]
    fn to_matrix_rotates_point_90_deg_y() {
        let t = Transform::from_rotation(Quat::from_rotation_y(FRAC_PI_2));
        let m = t.to_matrix();
        let result = m.transform_point3(Vec3::Z);
        // Z rotated 90° around Y => X
        assert!((result - Vec3::X).length() < 1e-5);
    }

    #[test]
    fn negative_scale_mirrors() {
        let t = Transform::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        let m = t.to_matrix();
        let result = m.transform_point3(Vec3::new(3.0, 4.0, 5.0));
        assert!((result - Vec3::new(-3.0, 4.0, 5.0)).length() < 1e-6);
    }

    #[test]
    fn zero_scale_collapses() {
        let t = Transform::from_scale(Vec3::ZERO);
        let m = t.to_matrix();
        let result = m.transform_point3(Vec3::new(100.0, 200.0, 300.0));
        assert!(result.length() < 1e-6);
    }

    #[test]
    fn combined_transform_applies_scale_then_rotation_then_translation() {
        let pos = Vec3::new(10.0, 0.0, 0.0);
        let rot = Quat::from_rotation_z(FRAC_PI_2);
        let scl = Vec3::new(2.0, 2.0, 2.0);
        let t = Transform::new(pos, rot, scl);
        let m = t.to_matrix();
        // Point (1,0,0) -> scale -> (2,0,0) -> rotate 90° Z -> (0,2,0) -> translate -> (10,2,0)
        let result = m.transform_point3(Vec3::X);
        assert!((result - Vec3::new(10.0, 2.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn identity_constant_fields() {
        assert_eq!(Transform::IDENTITY.position, Vec3::ZERO);
        assert_eq!(Transform::IDENTITY.rotation, Quat::IDENTITY);
        assert_eq!(Transform::IDENTITY.scale, Vec3::ONE);
    }

    #[test]
    fn clone_and_partial_eq() {
        let t = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(1.0),
            Vec3::new(4.0, 5.0, 6.0),
        );
        let t2 = t;
        assert_eq!(t, t2);
        assert_ne!(t, Transform::IDENTITY);
    }

    #[test]
    fn large_translation_values() {
        let t = Transform::from_position(Vec3::new(1e6, -1e6, 1e6));
        let m = t.to_matrix();
        let result = m.transform_point3(Vec3::ZERO);
        assert!((result - Vec3::new(1e6, -1e6, 1e6)).length() < 1.0);
    }
}
