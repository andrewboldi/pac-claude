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
}
