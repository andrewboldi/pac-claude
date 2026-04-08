use glam::Vec3;

/// Axis-aligned bounding box defined by minimum and maximum corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Create an AABB from explicit min/max corners.
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create an AABB from a center point and half-extents.
    #[inline]
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Test whether this AABB overlaps another (inclusive on boundaries).
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Test whether a point lies inside (or on the boundary of) this AABB.
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Return the smallest AABB that encloses both `self` and `other`.
    #[inline]
    pub fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_stores_min_max() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_eq!(a.min, Vec3::ZERO);
        assert_eq!(a.max, Vec3::ONE);
    }

    #[test]
    fn from_center_half_extents_roundtrip() {
        let center = Vec3::new(2.0, 3.0, 4.0);
        let half = Vec3::new(1.0, 1.0, 1.0);
        let a = Aabb::from_center_half_extents(center, half);
        assert_eq!(a.min, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a.max, Vec3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn intersects_overlapping() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        let b = Aabb::new(Vec3::ONE, Vec3::new(3.0, 3.0, 3.0));
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
    }

    #[test]
    fn intersects_touching_edge() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::ONE, Vec3::new(2.0, 2.0, 2.0));
        assert!(a.intersects(&b));
    }

    #[test]
    fn intersects_separated() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(!a.intersects(&b));
        assert!(!b.intersects(&a));
    }

    #[test]
    fn intersects_separated_single_axis() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        // Overlaps on Y and Z but separated on X
        let b = Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0));
        assert!(!a.intersects(&b));
    }

    #[test]
    fn contains_point_inside() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        assert!(a.contains_point(Vec3::ONE));
    }

    #[test]
    fn contains_point_on_boundary() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(a.contains_point(Vec3::ZERO));
        assert!(a.contains_point(Vec3::ONE));
        assert!(a.contains_point(Vec3::new(0.0, 0.5, 1.0)));
    }

    #[test]
    fn contains_point_outside() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(!a.contains_point(Vec3::new(2.0, 0.5, 0.5)));
        assert!(!a.contains_point(Vec3::new(-0.1, 0.5, 0.5)));
    }

    #[test]
    fn merge_produces_enclosing_aabb() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(-1.0, 2.0, 0.0), Vec3::new(0.5, 3.0, 4.0));
        let m = a.merge(&b);
        assert_eq!(m.min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(m.max, Vec3::new(1.0, 3.0, 4.0));
    }

    #[test]
    fn merge_with_self_is_identity() {
        let a = Aabb::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let m = a.merge(&a);
        assert_eq!(m, a);
    }

    #[test]
    fn merge_contained_aabb() {
        let outer = Aabb::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        let inner = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        let m = outer.merge(&inner);
        assert_eq!(m, outer);
    }

    #[test]
    fn merge_is_commutative() {
        let a = Aabb::new(Vec3::new(-1.0, 0.0, 2.0), Vec3::new(1.0, 3.0, 5.0));
        let b = Aabb::new(Vec3::new(0.0, -2.0, 1.0), Vec3::new(4.0, 1.0, 3.0));
        assert_eq!(a.merge(&b), b.merge(&a));
    }

    #[test]
    fn merge_negative_coordinates() {
        let a = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(-1.0, -1.0, -1.0));
        let b = Aabb::new(Vec3::new(-3.0, -3.0, -3.0), Vec3::new(0.0, 0.0, 0.0));
        let m = a.merge(&b);
        assert_eq!(m.min, Vec3::new(-5.0, -5.0, -5.0));
        assert_eq!(m.max, Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn new_with_negative_coordinates() {
        let a = Aabb::new(Vec3::new(-3.0, -2.0, -1.0), Vec3::new(-1.0, 0.0, 1.0));
        assert_eq!(a.min, Vec3::new(-3.0, -2.0, -1.0));
        assert_eq!(a.max, Vec3::new(-1.0, 0.0, 1.0));
    }

    #[test]
    fn from_center_half_extents_zero_half_extents() {
        let center = Vec3::new(5.0, 5.0, 5.0);
        let a = Aabb::from_center_half_extents(center, Vec3::ZERO);
        assert_eq!(a.min, center);
        assert_eq!(a.max, center);
    }

    #[test]
    fn from_center_half_extents_asymmetric() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(a.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn intersects_one_contains_other() {
        let outer = Aabb::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        let inner = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        assert!(outer.intersects(&inner));
        assert!(inner.intersects(&outer));
    }

    #[test]
    fn intersects_separated_y_axis_only() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 3.0, 1.0));
        assert!(!a.intersects(&b));
    }

    #[test]
    fn intersects_separated_z_axis_only() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(0.0, 0.0, 2.0), Vec3::new(1.0, 1.0, 3.0));
        assert!(!a.intersects(&b));
    }

    #[test]
    fn intersects_degenerate_point_aabb() {
        let point = Aabb::new(Vec3::splat(0.5), Vec3::splat(0.5));
        let box_aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(box_aabb.intersects(&point));
        assert!(point.intersects(&box_aabb));
    }

    #[test]
    fn intersects_degenerate_point_outside() {
        let point = Aabb::new(Vec3::splat(2.0), Vec3::splat(2.0));
        let box_aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(!box_aabb.intersects(&point));
    }

    #[test]
    fn intersects_degenerate_flat_plane() {
        let plane = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0));
        let box_aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::ONE);
        assert!(box_aabb.intersects(&plane));
    }

    #[test]
    fn contains_point_center() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        assert!(a.contains_point(Vec3::new(2.0, 2.0, 2.0)));
    }

    #[test]
    fn contains_point_just_outside_each_face() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let eps = 0.001;
        // Just outside each face
        assert!(!a.contains_point(Vec3::new(-eps, 0.5, 0.5)));
        assert!(!a.contains_point(Vec3::new(1.0 + eps, 0.5, 0.5)));
        assert!(!a.contains_point(Vec3::new(0.5, -eps, 0.5)));
        assert!(!a.contains_point(Vec3::new(0.5, 1.0 + eps, 0.5)));
        assert!(!a.contains_point(Vec3::new(0.5, 0.5, -eps)));
        assert!(!a.contains_point(Vec3::new(0.5, 0.5, 1.0 + eps)));
    }

    #[test]
    fn contains_point_at_all_corners() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        for x in [0.0_f32, 1.0] {
            for y in [0.0_f32, 1.0] {
                for z in [0.0_f32, 1.0] {
                    assert!(a.contains_point(Vec3::new(x, y, z)));
                }
            }
        }
    }

    #[test]
    fn contains_point_degenerate_point_aabb() {
        let a = Aabb::new(Vec3::splat(3.0), Vec3::splat(3.0));
        assert!(a.contains_point(Vec3::splat(3.0)));
        assert!(!a.contains_point(Vec3::splat(3.01)));
    }

    #[test]
    fn intersects_negative_space() {
        let a = Aabb::new(Vec3::new(-4.0, -4.0, -4.0), Vec3::new(-2.0, -2.0, -2.0));
        let b = Aabb::new(Vec3::new(-3.0, -3.0, -3.0), Vec3::new(-1.0, -1.0, -1.0));
        assert!(a.intersects(&b));
    }

    #[test]
    fn clone_and_partial_eq() {
        let a = Aabb::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let b = a;
        assert_eq!(a, b);
        let c = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_ne!(a, c);
    }
}
