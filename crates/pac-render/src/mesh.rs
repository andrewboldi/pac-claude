//! Mesh primitives with position, normal, and UV data.
//!
//! Each generator produces a [`Mesh`] containing [`Vertex3D`] vertices and
//! `u32` triangle-list indices, ready for GPU upload.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use crate::buffer::Vertex3D;

/// Triangle-list mesh stored as indexed vertex data.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Number of triangles in this mesh.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Axis-aligned unit cube (side length 1) centered at the origin.
    ///
    /// Each face has outward-facing normals and `[0,1]` UVs.
    /// 24 vertices (4 per face, unshared for correct normals), 36 indices.
    pub fn cube() -> Self {
        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        // (normal, tangent_u, tangent_v) for each face, used to build the quad.
        let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            // +X
            ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
            // -X
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            // +Y
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            // -Y
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            // +Z
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            // -Z
            ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ];

        for (normal, u_axis, v_axis) in &faces {
            let base = vertices.len() as u32;

            // Four corners: combinations of ±0.5 along u and v axes, offset by ±0.5 along normal.
            for &(su, sv) in &[(-0.5f32, -0.5f32), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
                let px = normal[0] * 0.5 + u_axis[0] * su + v_axis[0] * sv;
                let py = normal[1] * 0.5 + u_axis[1] * su + v_axis[1] * sv;
                let pz = normal[2] * 0.5 + u_axis[2] * su + v_axis[2] * sv;
                let tu = su + 0.5;
                let tv = sv + 0.5;
                vertices.push(Vertex3D {
                    position: [px, py, pz],
                    normal: *normal,
                    tex_coords: [tu, tv],
                });
            }

            // Two triangles per face (CCW winding when viewed from outside).
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self { vertices, indices }
    }

    /// UV sphere centered at the origin with radius 0.5.
    ///
    /// `sectors` controls longitude slices, `stacks` controls latitude slices.
    /// Minimum values are clamped to 3 sectors and 2 stacks.
    pub fn sphere(sectors: u32, stacks: u32) -> Self {
        let sectors = sectors.max(3);
        let stacks = stacks.max(2);

        let vert_count = ((stacks + 1) * (sectors + 1)) as usize;
        let idx_count = (stacks * sectors * 6) as usize;
        let mut vertices = Vec::with_capacity(vert_count);
        let mut indices = Vec::with_capacity(idx_count);

        let radius = 0.5f32;

        for i in 0..=stacks {
            let stack_angle = FRAC_PI_2 - (i as f32) * PI / (stacks as f32);
            let xy = radius * stack_angle.cos();
            let y = radius * stack_angle.sin();

            for j in 0..=sectors {
                let sector_angle = (j as f32) * TAU / (sectors as f32);
                let x = xy * sector_angle.cos();
                let z = xy * sector_angle.sin();

                let nx = stack_angle.cos() * sector_angle.cos();
                let ny = stack_angle.sin();
                let nz = stack_angle.cos() * sector_angle.sin();

                let u = j as f32 / sectors as f32;
                let v = i as f32 / stacks as f32;

                vertices.push(Vertex3D {
                    position: [x, y, z],
                    normal: [nx, ny, nz],
                    tex_coords: [u, v],
                });
            }
        }

        let ring = sectors + 1;
        for i in 0..stacks {
            for j in 0..sectors {
                let cur = i * ring + j;
                let next = cur + ring;

                if i != 0 {
                    indices.extend_from_slice(&[cur, next, cur + 1]);
                }
                if i != stacks - 1 {
                    indices.extend_from_slice(&[cur + 1, next, next + 1]);
                }
            }
        }

        Self { vertices, indices }
    }

    /// Flat XZ plane centered at the origin (Y = 0) with side length 1.
    ///
    /// `subdivisions` controls how many segments along each axis (minimum 1).
    /// Normal faces +Y. UVs span `[0,1]` across the plane.
    pub fn plane(subdivisions: u32) -> Self {
        let subdivisions = subdivisions.max(1);
        let verts_per_side = subdivisions + 1;
        let vert_count = (verts_per_side * verts_per_side) as usize;
        let idx_count = (subdivisions * subdivisions * 6) as usize;
        let mut vertices = Vec::with_capacity(vert_count);
        let mut indices = Vec::with_capacity(idx_count);

        let step = 1.0 / subdivisions as f32;

        for row in 0..verts_per_side {
            for col in 0..verts_per_side {
                let u = col as f32 * step;
                let v = row as f32 * step;
                let x = u - 0.5;
                let z = v - 0.5;
                vertices.push(Vertex3D {
                    position: [x, 0.0, z],
                    normal: [0.0, 1.0, 0.0],
                    tex_coords: [u, v],
                });
            }
        }

        for row in 0..subdivisions {
            for col in 0..subdivisions {
                let tl = row * verts_per_side + col;
                let tr = tl + 1;
                let bl = tl + verts_per_side;
                let br = bl + 1;
                indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
            }
        }

        Self { vertices, indices }
    }

    /// Cylinder centered at the origin with radius 0.5 and height 1 (along Y).
    ///
    /// `sectors` controls the number of radial slices (minimum 3).
    /// Includes top and bottom caps. Side UVs wrap horizontally; cap UVs are planar.
    pub fn cylinder(sectors: u32) -> Self {
        let sectors = sectors.max(3);
        let radius = 0.5f32;
        let half_h = 0.5f32;

        // Side: 2 rings × (sectors+1) verts, top cap: sectors+1 + 1, bottom cap: sectors+1 + 1
        let side_verts = 2 * (sectors + 1);
        let cap_verts = sectors + 2; // center + ring
        let vert_count = (side_verts + 2 * cap_verts) as usize;
        let side_idx = sectors * 6;
        let cap_idx = sectors * 3;
        let idx_count = (side_idx + 2 * cap_idx) as usize;

        let mut vertices = Vec::with_capacity(vert_count);
        let mut indices = Vec::with_capacity(idx_count);

        // ── Side ──
        for i in 0..=sectors {
            let angle = (i as f32) * TAU / (sectors as f32);
            let (sin_a, cos_a) = angle.sin_cos();
            let nx = cos_a;
            let nz = sin_a;
            let u = i as f32 / sectors as f32;

            // Bottom ring vertex
            vertices.push(Vertex3D {
                position: [radius * cos_a, -half_h, radius * sin_a],
                normal: [nx, 0.0, nz],
                tex_coords: [u, 1.0],
            });
            // Top ring vertex
            vertices.push(Vertex3D {
                position: [radius * cos_a, half_h, radius * sin_a],
                normal: [nx, 0.0, nz],
                tex_coords: [u, 0.0],
            });
        }

        for i in 0..sectors {
            let b = i * 2;
            let t = b + 1;
            let b_next = b + 2;
            let t_next = b + 3;
            // Two triangles per side quad
            indices.extend_from_slice(&[b, b_next, t, t, b_next, t_next]);
        }

        // ── Top cap ──
        let top_center = vertices.len() as u32;
        vertices.push(Vertex3D {
            position: [0.0, half_h, 0.0],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [0.5, 0.5],
        });
        for i in 0..=sectors {
            let angle = (i as f32) * TAU / (sectors as f32);
            let (sin_a, cos_a) = angle.sin_cos();
            vertices.push(Vertex3D {
                position: [radius * cos_a, half_h, radius * sin_a],
                normal: [0.0, 1.0, 0.0],
                tex_coords: [cos_a * 0.5 + 0.5, sin_a * 0.5 + 0.5],
            });
        }
        for i in 0..sectors {
            indices.extend_from_slice(&[top_center, top_center + 1 + i, top_center + 2 + i]);
        }

        // ── Bottom cap ──
        let bot_center = vertices.len() as u32;
        vertices.push(Vertex3D {
            position: [0.0, -half_h, 0.0],
            normal: [0.0, -1.0, 0.0],
            tex_coords: [0.5, 0.5],
        });
        for i in 0..=sectors {
            let angle = (i as f32) * TAU / (sectors as f32);
            let (sin_a, cos_a) = angle.sin_cos();
            vertices.push(Vertex3D {
                position: [radius * cos_a, -half_h, radius * sin_a],
                normal: [0.0, -1.0, 0.0],
                tex_coords: [cos_a * 0.5 + 0.5, sin_a * 0.5 + 0.5],
            });
        }
        for i in 0..sectors {
            // Reverse winding so normal faces outward (-Y)
            indices.extend_from_slice(&[bot_center, bot_center + 2 + i, bot_center + 1 + i]);
        }

        Self { vertices, indices }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn is_unit_length(n: [f32; 3]) -> bool {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        approx_eq(len, 1.0)
    }

    fn validate_indices(mesh: &Mesh) {
        let n = mesh.vertices.len() as u32;
        for &idx in &mesh.indices {
            assert!(idx < n, "index {idx} out of bounds (vertex count {n})");
        }
        assert_eq!(mesh.indices.len() % 3, 0, "indices not a multiple of 3");
    }

    fn uvs_in_range(mesh: &Mesh) {
        for v in &mesh.vertices {
            assert!(
                v.tex_coords[0] >= -1e-5 && v.tex_coords[0] <= 1.0 + 1e-5,
                "U out of range: {}",
                v.tex_coords[0]
            );
            assert!(
                v.tex_coords[1] >= -1e-5 && v.tex_coords[1] <= 1.0 + 1e-5,
                "V out of range: {}",
                v.tex_coords[1]
            );
        }
    }

    // ── Cube ──────────────────────────────────────────────────────

    #[test]
    fn cube_vertex_count() {
        let m = Mesh::cube();
        assert_eq!(m.vertices.len(), 24);
    }

    #[test]
    fn cube_index_count() {
        let m = Mesh::cube();
        assert_eq!(m.indices.len(), 36);
    }

    #[test]
    fn cube_triangle_count() {
        let m = Mesh::cube();
        assert_eq!(m.triangle_count(), 12);
    }

    #[test]
    fn cube_indices_valid() {
        validate_indices(&Mesh::cube());
    }

    #[test]
    fn cube_normals_unit_length() {
        for v in &Mesh::cube().vertices {
            assert!(is_unit_length(v.normal), "non-unit normal: {:?}", v.normal);
        }
    }

    #[test]
    fn cube_uvs_in_range() {
        uvs_in_range(&Mesh::cube());
    }

    #[test]
    fn cube_positions_within_half_unit() {
        for v in &Mesh::cube().vertices {
            for &c in &v.position {
                assert!(
                    approx_eq(c.abs(), 0.5),
                    "cube position component not ±0.5: {c}"
                );
            }
        }
    }

    // ── Sphere ────────────────────────────────────────────────────

    #[test]
    fn sphere_indices_valid() {
        validate_indices(&Mesh::sphere(16, 8));
    }

    #[test]
    fn sphere_normals_unit_length() {
        for v in &Mesh::sphere(16, 8).vertices {
            assert!(is_unit_length(v.normal), "non-unit normal: {:?}", v.normal);
        }
    }

    #[test]
    fn sphere_uvs_in_range() {
        uvs_in_range(&Mesh::sphere(16, 8));
    }

    #[test]
    fn sphere_radius_is_half() {
        for v in &Mesh::sphere(16, 8).vertices {
            let r = (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt();
            assert!(approx_eq(r, 0.5), "vertex radius {r} != 0.5");
        }
    }

    #[test]
    fn sphere_minimum_clamps() {
        let m = Mesh::sphere(1, 1); // should clamp to 3 sectors, 2 stacks
        assert!(m.vertices.len() >= 12); // (2+1)*(3+1)
        validate_indices(&m);
    }

    #[test]
    fn sphere_vertex_count_formula() {
        let (sec, stk) = (16u32, 8u32);
        let m = Mesh::sphere(sec, stk);
        assert_eq!(m.vertices.len(), ((stk + 1) * (sec + 1)) as usize);
    }

    // ── Plane ─────────────────────────────────────────────────────

    #[test]
    fn plane_indices_valid() {
        validate_indices(&Mesh::plane(4));
    }

    #[test]
    fn plane_vertex_count() {
        let m = Mesh::plane(4);
        assert_eq!(m.vertices.len(), 25); // 5×5
    }

    #[test]
    fn plane_index_count() {
        let m = Mesh::plane(4);
        assert_eq!(m.indices.len(), 4 * 4 * 6); // 96
    }

    #[test]
    fn plane_normals_point_up() {
        for v in &Mesh::plane(2).vertices {
            assert_eq!(v.normal, [0.0, 1.0, 0.0]);
        }
    }

    #[test]
    fn plane_y_is_zero() {
        for v in &Mesh::plane(3).vertices {
            assert!(approx_eq(v.position[1], 0.0));
        }
    }

    #[test]
    fn plane_uvs_in_range() {
        uvs_in_range(&Mesh::plane(4));
    }

    #[test]
    fn plane_positions_within_half() {
        for v in &Mesh::plane(4).vertices {
            assert!(v.position[0] >= -0.5 - 1e-5 && v.position[0] <= 0.5 + 1e-5);
            assert!(v.position[2] >= -0.5 - 1e-5 && v.position[2] <= 0.5 + 1e-5);
        }
    }

    #[test]
    fn plane_minimum_clamps() {
        let m = Mesh::plane(0); // should clamp to 1
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.indices.len(), 6);
        validate_indices(&m);
    }

    // ── Cylinder ──────────────────────────────────────────────────

    #[test]
    fn cylinder_indices_valid() {
        validate_indices(&Mesh::cylinder(16));
    }

    #[test]
    fn cylinder_normals_unit_length() {
        for v in &Mesh::cylinder(16).vertices {
            assert!(is_unit_length(v.normal), "non-unit normal: {:?}", v.normal);
        }
    }

    #[test]
    fn cylinder_uvs_in_range() {
        uvs_in_range(&Mesh::cylinder(16));
    }

    #[test]
    fn cylinder_height_is_one() {
        for v in &Mesh::cylinder(16).vertices {
            assert!(
                v.position[1] >= -0.5 - 1e-5 && v.position[1] <= 0.5 + 1e-5,
                "y out of range: {}",
                v.position[1]
            );
        }
    }

    #[test]
    fn cylinder_radius_within_half() {
        for v in &Mesh::cylinder(16).vertices {
            let r = (v.position[0].powi(2) + v.position[2].powi(2)).sqrt();
            assert!(r <= 0.5 + 1e-5, "radius {r} > 0.5");
        }
    }

    #[test]
    fn cylinder_minimum_clamps() {
        let m = Mesh::cylinder(1); // should clamp to 3
        validate_indices(&m);
        assert!(m.triangle_count() >= 8); // 3 side quads (6 tri) + 3 top + 3 bot = 12
    }

    #[test]
    fn cylinder_has_caps() {
        let m = Mesh::cylinder(8);
        // Should have vertices with normal [0,1,0] (top cap) and [0,-1,0] (bottom cap)
        let has_top = m.vertices.iter().any(|v| v.normal == [0.0, 1.0, 0.0]);
        let has_bot = m.vertices.iter().any(|v| v.normal == [0.0, -1.0, 0.0]);
        assert!(has_top, "missing top cap vertices");
        assert!(has_bot, "missing bottom cap vertices");
    }
}
