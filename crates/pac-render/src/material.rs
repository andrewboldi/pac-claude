//! Material system: diffuse color/texture, specular, uniform buffer + bind group.
//!
//! A [`Material`] bundles surface appearance properties (diffuse color, specular
//! highlights) with an optional diffuse [`Texture`], a GPU uniform buffer, and a
//! ready-to-bind [`wgpu::BindGroup`].
//!
//! Bind group layout (one group per material):
//!
//! | Binding | Resource              | Stage    |
//! |---------|-----------------------|----------|
//! | 0       | `MaterialUniforms`    | Fragment |
//! | 1       | Diffuse texture view  | Fragment |
//! | 2       | Diffuse sampler       | Fragment |

use crate::buffer::UniformBuffer;
use crate::texture::Texture;

// ── MaterialUniforms ──────────────────────────────────────────────────────

/// GPU-side material uniform data.
///
/// Memory layout (32 bytes, `#[repr(C)]`):
///
/// | Field               | Offset | Format    |
/// |---------------------|--------|-----------|
/// | `diffuse_color`     | 0      | Float32x4 |
/// | `specular_shininess`| 16     | Float32x4 |
///
/// `diffuse_color.w` acts as a flag: `1.0` when a diffuse texture is bound
/// (the shader should sample the texture), `0.0` when only the color is used.
///
/// `specular_shininess.w` holds the Phong shininess exponent.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniforms {
    /// Diffuse color (RGB) + has-diffuse-texture flag (A: 0.0 or 1.0).
    pub diffuse_color: [f32; 4],
    /// Specular color (RGB) + shininess exponent (A).
    pub specular_shininess: [f32; 4],
}

impl MaterialUniforms {
    /// Create material uniforms from individual components.
    #[inline]
    pub fn new(
        diffuse: [f32; 3],
        specular: [f32; 3],
        shininess: f32,
        has_texture: bool,
    ) -> Self {
        Self {
            diffuse_color: [
                diffuse[0],
                diffuse[1],
                diffuse[2],
                if has_texture { 1.0 } else { 0.0 },
            ],
            specular_shininess: [specular[0], specular[1], specular[2], shininess],
        }
    }
}

impl Default for MaterialUniforms {
    /// Default material: medium grey diffuse, white specular, shininess 32.
    fn default() -> Self {
        Self::new([0.7, 0.7, 0.7], [0.5, 0.5, 0.5], 32.0, false)
    }
}

// ── Material ──────────────────────────────────────────────────────────────

/// Material with diffuse color/texture, specular properties, and GPU bind group.
///
/// Materials that have no diffuse texture use an internal 1×1 white fallback so
/// the bind group layout stays uniform across all materials.
pub struct Material {
    diffuse_texture: Texture,
    uniform_buffer: UniformBuffer<MaterialUniforms>,
    bind_group: wgpu::BindGroup,
}

impl Material {
    /// Create the bind group layout shared by all materials.
    ///
    /// Binding 0: `MaterialUniforms` (uniform buffer, fragment stage).
    /// Binding 1: Diffuse texture view (2D float, fragment stage).
    /// Binding 2: Diffuse sampler (filtering, fragment stage).
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    /// Create a material with a solid diffuse color (no texture).
    ///
    /// A 1×1 white fallback texture is bound so the bind group layout stays
    /// consistent. The shader uses `diffuse_color.w == 0.0` to skip sampling.
    pub fn from_color(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        diffuse: [f32; 3],
        specular: [f32; 3],
        shininess: f32,
    ) -> Self {
        let uniforms = MaterialUniforms::new(diffuse, specular, shininess, false);
        let fallback = Self::white_texture(device, queue);
        Self::build(device, layout, uniforms, fallback)
    }

    /// Create a material with a diffuse texture.
    ///
    /// The diffuse color is set to white so the texture color passes through
    /// unmodified. Set `specular` and `shininess` for highlight control.
    pub fn from_texture(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        diffuse_texture: Texture,
        specular: [f32; 3],
        shininess: f32,
    ) -> Self {
        let uniforms = MaterialUniforms::new([1.0, 1.0, 1.0], specular, shininess, true);
        Self::build(device, layout, uniforms, diffuse_texture)
    }

    /// Create a default grey material.
    pub fn default_material(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let uniforms = MaterialUniforms::default();
        let fallback = Self::white_texture(device, queue);
        Self::build(device, layout, uniforms, fallback)
    }

    /// The material's bind group, ready for use in a render pass.
    #[inline]
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// The bound diffuse texture (may be a 1×1 white fallback).
    #[inline]
    pub fn diffuse_texture(&self) -> &Texture {
        &self.diffuse_texture
    }

    /// Overwrite the material uniform buffer with new data.
    #[inline]
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &MaterialUniforms) {
        self.uniform_buffer.write(queue, uniforms);
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn build(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniforms: MaterialUniforms,
        diffuse_texture: Texture,
    ) -> Self {
        let uniform_buffer = UniformBuffer::new(device, "material_uniforms", &uniforms);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
        });

        Self {
            diffuse_texture,
            uniform_buffer,
            bind_group,
        }
    }

    fn white_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> Texture {
        Texture::from_rgba8(device, queue, &[255, 255, 255, 255], 1, 1, "white_fallback")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use std::mem;

    // ── MaterialUniforms layout ──────────────────────────────────

    #[test]
    fn uniforms_size_is_32_bytes() {
        assert_eq!(mem::size_of::<MaterialUniforms>(), 32);
    }

    #[test]
    fn uniforms_alignment_is_4() {
        assert_eq!(mem::align_of::<MaterialUniforms>(), 4);
    }

    // ── MaterialUniforms constructor ─────────────────────────────

    #[test]
    fn new_stores_diffuse_color() {
        let u = MaterialUniforms::new([0.1, 0.2, 0.3], [0.0; 3], 1.0, false);
        assert_eq!(u.diffuse_color[0], 0.1);
        assert_eq!(u.diffuse_color[1], 0.2);
        assert_eq!(u.diffuse_color[2], 0.3);
    }

    #[test]
    fn new_stores_specular_color() {
        let u = MaterialUniforms::new([0.0; 3], [0.4, 0.5, 0.6], 1.0, false);
        assert_eq!(u.specular_shininess[0], 0.4);
        assert_eq!(u.specular_shininess[1], 0.5);
        assert_eq!(u.specular_shininess[2], 0.6);
    }

    #[test]
    fn new_stores_shininess() {
        let u = MaterialUniforms::new([0.0; 3], [0.0; 3], 64.0, false);
        assert_eq!(u.specular_shininess[3], 64.0);
    }

    #[test]
    fn has_texture_flag_false() {
        let u = MaterialUniforms::new([0.0; 3], [0.0; 3], 1.0, false);
        assert_eq!(u.diffuse_color[3], 0.0);
    }

    #[test]
    fn has_texture_flag_true() {
        let u = MaterialUniforms::new([0.0; 3], [0.0; 3], 1.0, true);
        assert_eq!(u.diffuse_color[3], 1.0);
    }

    // ── MaterialUniforms default ─────────────────────────────────

    #[test]
    fn default_diffuse_is_grey() {
        let u = MaterialUniforms::default();
        assert_eq!(u.diffuse_color[0], 0.7);
        assert_eq!(u.diffuse_color[1], 0.7);
        assert_eq!(u.diffuse_color[2], 0.7);
    }

    #[test]
    fn default_has_no_texture() {
        let u = MaterialUniforms::default();
        assert_eq!(u.diffuse_color[3], 0.0);
    }

    #[test]
    fn default_specular_is_half_white() {
        let u = MaterialUniforms::default();
        assert_eq!(u.specular_shininess[0], 0.5);
        assert_eq!(u.specular_shininess[1], 0.5);
        assert_eq!(u.specular_shininess[2], 0.5);
    }

    #[test]
    fn default_shininess_is_32() {
        let u = MaterialUniforms::default();
        assert_eq!(u.specular_shininess[3], 32.0);
    }

    // ── bytemuck safety ──────────────────────────────────────────

    #[test]
    fn uniforms_round_trip_through_bytes() {
        let u = MaterialUniforms::new([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], 64.0, true);
        let bytes = bytemuck::bytes_of(&u);
        let back: &MaterialUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(&u, back);
    }

    #[test]
    fn uniforms_cast_slice_round_trips() {
        let mats = [
            MaterialUniforms::default(),
            MaterialUniforms::new([1.0, 0.0, 0.0], [1.0, 1.0, 1.0], 128.0, true),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&mats);
        assert_eq!(bytes.len(), 64); // 2 * 32
        let back: &[MaterialUniforms] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &mats);
    }

    #[test]
    fn zeroed_uniforms_are_all_zero() {
        let u = MaterialUniforms::zeroed();
        assert_eq!(u.diffuse_color, [0.0; 4]);
        assert_eq!(u.specular_shininess, [0.0; 4]);
    }
}
