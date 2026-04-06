//! Light types and GPU uniform buffer manager for Phong shading.
//!
//! Provides [`PointLight`] and [`DirectionalLight`] CPU-side types, a
//! [`LightUniforms`] struct that maps directly to the WGSL uniform layout,
//! and a [`LightManager`] that owns the GPU buffer and bind group.

use crate::buffer::UniformBuffer;
use crate::pipeline::{uniform_bind_group, uniform_bind_group_layout};
use glam::Vec3;

/// Maximum number of point lights supported by the uniform buffer.
///
/// This must match the array size in `phong.wgsl`.
pub const MAX_POINT_LIGHTS: usize = 4;

// ── GPU-side raw structs (match WGSL layout exactly) ────────────────────

/// GPU representation of a directional light.
///
/// Layout: 32 bytes (2 × `vec4<f32>`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectionalLightRaw {
    /// Normalized direction *toward* the light (xyz), w unused.
    pub direction: [f32; 4],
    /// RGB color (xyz), intensity (w).
    pub color: [f32; 4],
}

/// GPU representation of a point light.
///
/// Layout: 48 bytes (3 × `vec4<f32>`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLightRaw {
    /// World-space position (xyz), w unused.
    pub position: [f32; 4],
    /// RGB color (xyz), intensity (w).
    pub color: [f32; 4],
    /// Attenuation: constant (x), linear (y), quadratic (z), w unused.
    pub attenuation: [f32; 4],
}

/// Uniform buffer data sent to the GPU each frame.
///
/// Must match the `LightUniforms` struct in `phong.wgsl` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniforms {
    /// Ambient light color (xyz), w unused.
    pub ambient: [f32; 4],
    /// The single directional light.
    pub directional: DirectionalLightRaw,
    /// Fixed-size array of point lights.
    pub point_lights: [PointLightRaw; MAX_POINT_LIGHTS],
    /// Number of active point lights (0..=MAX_POINT_LIGHTS).
    pub num_point_lights: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 3],
}

// ── CPU-side ergonomic types ────────────────────────────────────────────

/// A directional light (e.g. the sun).
///
/// `direction` points *from* the surface *toward* the light source. The shader
/// uses this directly for N·L calculations.
#[derive(Clone, Debug)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.3, 1.0, 0.5).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
        }
    }
}

impl DirectionalLight {
    fn to_raw(&self) -> DirectionalLightRaw {
        let dir = self.direction.normalize();
        DirectionalLightRaw {
            direction: [dir.x, dir.y, dir.z, 0.0],
            color: [self.color.x, self.color.y, self.color.z, self.intensity],
        }
    }
}

/// A point light that attenuates with distance.
///
/// Attenuation follows `1 / (constant + linear*d + quadratic*d²)`.
#[derive(Clone, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub constant_att: f32,
    pub linear_att: f32,
    pub quadratic_att: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1.0,
            constant_att: 1.0,
            linear_att: 0.09,
            quadratic_att: 0.032,
        }
    }
}

impl PointLight {
    fn to_raw(&self) -> PointLightRaw {
        PointLightRaw {
            position: [self.position.x, self.position.y, self.position.z, 0.0],
            color: [self.color.x, self.color.y, self.color.z, self.intensity],
            attenuation: [self.constant_att, self.linear_att, self.quadratic_att, 0.0],
        }
    }
}

// ── LightManager ────────────────────────────────────────────────────────

/// Manages the scene lights and their GPU uniform buffer.
///
/// Owns the [`UniformBuffer<LightUniforms>`], the bind group, and the
/// bind group layout. Call [`Self::write`] each frame after modifying
/// lights to upload the data to the GPU.
pub struct LightManager {
    pub ambient: Vec3,
    pub directional: DirectionalLight,
    pub point_lights: Vec<PointLight>,
    buffer: UniformBuffer<LightUniforms>,
    bind_group: wgpu::BindGroup,
    layout: wgpu::BindGroupLayout,
}

impl LightManager {
    /// Create a new light manager with default lighting.
    ///
    /// Defaults: white ambient at 10%, a white directional light from
    /// the upper-right-front, and no point lights.
    pub fn new(device: &wgpu::Device) -> Self {
        let ambient = Vec3::splat(0.1);
        let directional = DirectionalLight::default();
        let point_lights = Vec::new();

        let uniforms = Self::build_uniforms(ambient, &directional, &point_lights);

        let buffer = UniformBuffer::new(device, "light_uniforms", &uniforms);
        let layout = uniform_bind_group_layout(
            device,
            "light_bind_group_layout",
            wgpu::ShaderStages::FRAGMENT,
        );
        let bind_group = uniform_bind_group(device, "light_bind_group", &layout, buffer.buffer());

        Self {
            ambient,
            directional,
            point_lights,
            buffer,
            bind_group,
            layout,
        }
    }

    /// Add a point light. Returns `None` if already at [`MAX_POINT_LIGHTS`].
    pub fn add_point_light(&mut self, light: PointLight) -> Option<usize> {
        if self.point_lights.len() >= MAX_POINT_LIGHTS {
            return None;
        }
        let idx = self.point_lights.len();
        self.point_lights.push(light);
        Some(idx)
    }

    /// Remove all point lights.
    pub fn clear_point_lights(&mut self) {
        self.point_lights.clear();
    }

    /// Upload current light state to the GPU.
    ///
    /// Call once per frame, after any modifications to lights.
    pub fn write(&self, queue: &wgpu::Queue) {
        let uniforms = Self::build_uniforms(self.ambient, &self.directional, &self.point_lights);
        self.buffer.write(queue, &uniforms);
    }

    /// The bind group layout (needed when creating the render pipeline).
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// The bind group (set on the render pass each frame).
    #[inline]
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    fn build_uniforms(
        ambient: Vec3,
        directional: &DirectionalLight,
        point_lights: &[PointLight],
    ) -> LightUniforms {
        let mut raw_points = [PointLightRaw::zeroed(); MAX_POINT_LIGHTS];
        let count = point_lights.len().min(MAX_POINT_LIGHTS);
        for (i, pl) in point_lights.iter().take(MAX_POINT_LIGHTS).enumerate() {
            raw_points[i] = pl.to_raw();
        }

        LightUniforms {
            ambient: [ambient.x, ambient.y, ambient.z, 0.0],
            directional: directional.to_raw(),
            point_lights: raw_points,
            num_point_lights: count as u32,
            _pad: [0; 3],
        }
    }
}

impl PointLightRaw {
    const fn zeroed() -> Self {
        Self {
            position: [0.0; 4],
            color: [0.0; 4],
            attenuation: [0.0; 4],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    // ── Layout / alignment ──────────────────────────────────────────

    #[test]
    fn directional_light_raw_size() {
        assert_eq!(mem::size_of::<DirectionalLightRaw>(), 32);
    }

    #[test]
    fn point_light_raw_size() {
        assert_eq!(mem::size_of::<PointLightRaw>(), 48);
    }

    #[test]
    fn light_uniforms_size_is_256() {
        // 16 (ambient) + 32 (directional) + 192 (4 * 48 point lights)
        // + 4 (count) + 12 (pad) = 256
        assert_eq!(mem::size_of::<LightUniforms>(), 256);
    }

    #[test]
    fn light_uniforms_is_16_aligned() {
        assert_eq!(mem::size_of::<LightUniforms>() % 16, 0);
    }

    // ── bytemuck safety ─────────────────────────────────────────────

    #[test]
    fn light_uniforms_round_trips_through_bytes() {
        let u = LightUniforms {
            ambient: [0.1, 0.1, 0.1, 0.0],
            directional: DirectionalLightRaw {
                direction: [0.0, 1.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            point_lights: [PointLightRaw::zeroed(); MAX_POINT_LIGHTS],
            num_point_lights: 0,
            _pad: [0; 3],
        };
        let bytes = bytemuck::bytes_of(&u);
        let back: &LightUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(back.ambient, u.ambient);
        assert_eq!(back.num_point_lights, 0);
    }

    // ── DirectionalLight ────────────────────────────────────────────

    #[test]
    fn directional_light_default_is_normalized() {
        let dl = DirectionalLight::default();
        let len = dl.direction.length();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn directional_light_to_raw_normalizes() {
        let dl = DirectionalLight {
            direction: Vec3::new(0.0, 2.0, 0.0),
            color: Vec3::new(1.0, 0.5, 0.0),
            intensity: 0.8,
        };
        let raw = dl.to_raw();
        assert!((raw.direction[1] - 1.0).abs() < 1e-5);
        assert!((raw.color[3] - 0.8).abs() < 1e-5);
    }

    // ── PointLight ──────────────────────────────────────────────────

    #[test]
    fn point_light_to_raw_packs_correctly() {
        let pl = PointLight {
            position: Vec3::new(1.0, 2.0, 3.0),
            color: Vec3::new(0.5, 0.5, 0.5),
            intensity: 2.0,
            constant_att: 1.0,
            linear_att: 0.1,
            quadratic_att: 0.01,
        };
        let raw = pl.to_raw();
        assert_eq!(raw.position[0], 1.0);
        assert_eq!(raw.position[1], 2.0);
        assert_eq!(raw.position[2], 3.0);
        assert_eq!(raw.color[3], 2.0); // intensity in w
        assert_eq!(raw.attenuation[0], 1.0);
        assert_eq!(raw.attenuation[1], 0.1);
        assert!((raw.attenuation[2] - 0.01).abs() < 1e-6);
    }

    // ── build_uniforms ──────────────────────────────────────────────

    #[test]
    fn build_uniforms_with_no_point_lights() {
        let u = LightManager::build_uniforms(
            Vec3::splat(0.2),
            &DirectionalLight::default(),
            &[],
        );
        assert_eq!(u.num_point_lights, 0);
        assert!((u.ambient[0] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn build_uniforms_clamps_to_max() {
        let lights: Vec<PointLight> = (0..MAX_POINT_LIGHTS + 2)
            .map(|i| PointLight {
                position: Vec3::new(i as f32, 0.0, 0.0),
                ..PointLight::default()
            })
            .collect();
        let u = LightManager::build_uniforms(Vec3::ZERO, &DirectionalLight::default(), &lights);
        assert_eq!(u.num_point_lights, MAX_POINT_LIGHTS as u32);
    }

    #[test]
    fn build_uniforms_preserves_point_light_order() {
        let lights = vec![
            PointLight {
                position: Vec3::new(1.0, 0.0, 0.0),
                ..PointLight::default()
            },
            PointLight {
                position: Vec3::new(2.0, 0.0, 0.0),
                ..PointLight::default()
            },
        ];
        let u = LightManager::build_uniforms(Vec3::ZERO, &DirectionalLight::default(), &lights);
        assert_eq!(u.num_point_lights, 2);
        assert_eq!(u.point_lights[0].position[0], 1.0);
        assert_eq!(u.point_lights[1].position[0], 2.0);
    }
}
