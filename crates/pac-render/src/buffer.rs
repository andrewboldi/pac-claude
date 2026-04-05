//! GPU buffer abstractions wrapping `wgpu::Buffer` with type-safe `bytemuck` serialization.

use std::marker::PhantomData;
use std::mem;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

// ── Vertex trait ─────────────────────────────────────────────────────────

/// Trait for vertex types that can describe their GPU buffer layout.
///
/// Implementors must derive [`bytemuck::Pod`] and [`bytemuck::Zeroable`] to
/// ensure safe CPU-to-GPU data transfer.
pub trait Vertex: Pod {
    /// Returns the vertex buffer layout for use in render pipeline descriptors.
    fn layout() -> wgpu::VertexBufferLayout<'static>;
}

// ── VertexBuffer ─────────────────────────────────────────────────────────

/// Type-safe GPU vertex buffer.
///
/// Wraps a [`wgpu::Buffer`] created with [`wgpu::BufferUsages::VERTEX`] and
/// tracks the vertex count for draw calls.
pub struct VertexBuffer<V: Vertex> {
    buffer: wgpu::Buffer,
    count: u32,
    _marker: PhantomData<V>,
}

impl<V: Vertex> VertexBuffer<V> {
    /// Create a vertex buffer initialized with `vertices`.
    pub fn new(device: &wgpu::Device, label: &str, vertices: &[V]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            buffer,
            count: vertices.len() as u32,
            _marker: PhantomData,
        }
    }

    /// The underlying `wgpu::Buffer`.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Number of vertices stored in this buffer.
    #[inline]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Vertex buffer layout for pipeline creation.
    #[inline]
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        V::layout()
    }

    /// Full buffer slice, for binding in a render pass.
    #[inline]
    pub fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(..)
    }
}

// ── UniformBuffer ────────────────────────────────────────────────────────

/// Type-safe GPU uniform buffer.
///
/// Wraps a [`wgpu::Buffer`] created with [`wgpu::BufferUsages::UNIFORM`] |
/// [`wgpu::BufferUsages::COPY_DST`] so it can be updated each frame via
/// [`wgpu::Queue::write_buffer`].
pub struct UniformBuffer<U: Pod> {
    buffer: wgpu::Buffer,
    _marker: PhantomData<U>,
}

impl<U: Pod> UniformBuffer<U> {
    /// Create a uniform buffer initialized with `data`.
    pub fn new(device: &wgpu::Device, label: &str, data: &U) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    /// Overwrite the entire buffer with new data.
    ///
    /// Typically called once per frame to upload updated transforms or other
    /// per-frame uniforms.
    #[inline]
    pub fn write(&self, queue: &wgpu::Queue, data: &U) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(data));
    }

    /// The underlying `wgpu::Buffer`.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

// ── Standard vertex types ────────────────────────────────────────────────

/// 3D vertex with position, normal, and texture coordinates.
///
/// Memory layout (32 bytes, `#[repr(C)]`):
///
/// | Field        | Offset | Format    |
/// |--------------|--------|-----------|
/// | `position`   | 0      | Float32x3 |
/// | `normal`     | 12     | Float32x3 |
/// | `tex_coords` | 24     | Float32x2 |
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

const VERTEX3D_ATTRS: [wgpu::VertexAttribute; 3] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 12,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 24,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32x2,
    },
];

impl Vertex for Vertex3D {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX3D_ATTRS,
        }
    }
}

// ── InstanceData ──────────────────────────────────────────────────────────

/// Per-instance data for instanced rendering.
///
/// Contains a 4×4 column-major model matrix matching `glam::Mat4` layout.
///
/// Memory layout (64 bytes, `#[repr(C)]`):
///
/// | Field       | Offset | Format    |
/// |-------------|--------|-----------|
/// | `model[0]`  | 0      | Float32x4 |
/// | `model[1]`  | 16     | Float32x4 |
/// | `model[2]`  | 32     | Float32x4 |
/// | `model[3]`  | 48     | Float32x4 |
///
/// Shader locations 3–6 (following [`Vertex3D`] at 0–2).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],
}

impl InstanceData {
    /// Build from a `glam::Mat4`.
    #[inline]
    pub fn from_mat4(mat: glam::Mat4) -> Self {
        Self {
            model: mat.to_cols_array_2d(),
        }
    }

    /// Identity instance (no transform).
    pub const IDENTITY: Self = Self {
        model: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Vertex buffer layout for the instance slot (step_mode = Instance).
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_DATA_ATTRS,
        }
    }
}

const INSTANCE_DATA_ATTRS: [wgpu::VertexAttribute; 4] = [
    // Column 0
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 3,
        format: wgpu::VertexFormat::Float32x4,
    },
    // Column 1
    wgpu::VertexAttribute {
        offset: 16,
        shader_location: 4,
        format: wgpu::VertexFormat::Float32x4,
    },
    // Column 2
    wgpu::VertexAttribute {
        offset: 32,
        shader_location: 5,
        format: wgpu::VertexFormat::Float32x4,
    },
    // Column 3
    wgpu::VertexAttribute {
        offset: 48,
        shader_location: 6,
        format: wgpu::VertexFormat::Float32x4,
    },
];

// ── InstanceBuffer ────────────────────────────────────────────────────────

/// GPU buffer holding per-instance data for instanced draw calls.
///
/// Created with [`wgpu::BufferUsages::VERTEX`] | [`wgpu::BufferUsages::COPY_DST`]
/// so instance transforms can be updated each frame via [`Self::write`].
pub struct InstanceBuffer {
    buffer: wgpu::Buffer,
    count: u32,
}

impl InstanceBuffer {
    /// Create an instance buffer initialized with `instances`.
    pub fn new(device: &wgpu::Device, label: &str, instances: &[InstanceData]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            buffer,
            count: instances.len() as u32,
        }
    }

    /// Overwrite the buffer with new instance data.
    ///
    /// If `instances.len()` differs from the current count, the buffer is
    /// recreated on `device`. Otherwise only a queue write is performed.
    pub fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[InstanceData],
    ) {
        let new_count = instances.len() as u32;
        if new_count != self.count {
            self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("instance_buffer"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.count = new_count;
        } else {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(instances));
        }
    }

    /// The underlying `wgpu::Buffer`.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Number of instances stored in this buffer.
    #[inline]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Full buffer slice, for binding in a render pass.
    #[inline]
    pub fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(..)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Vertex3D layout ──────────────────────────────────────────

    #[test]
    fn vertex3d_size_is_32_bytes() {
        assert_eq!(mem::size_of::<Vertex3D>(), 32);
    }

    #[test]
    fn vertex3d_layout_stride_matches_size() {
        let layout = Vertex3D::layout();
        assert_eq!(layout.array_stride, 32);
    }

    #[test]
    fn vertex3d_layout_step_mode_is_vertex() {
        let layout = Vertex3D::layout();
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
    }

    #[test]
    fn vertex3d_layout_has_three_attributes() {
        let layout = Vertex3D::layout();
        assert_eq!(layout.attributes.len(), 3);
    }

    #[test]
    fn vertex3d_position_attribute() {
        let attrs = Vertex3D::layout().attributes;
        assert_eq!(attrs[0].offset, 0);
        assert_eq!(attrs[0].shader_location, 0);
        assert_eq!(attrs[0].format, wgpu::VertexFormat::Float32x3);
    }

    #[test]
    fn vertex3d_normal_attribute() {
        let attrs = Vertex3D::layout().attributes;
        assert_eq!(attrs[1].offset, 12);
        assert_eq!(attrs[1].shader_location, 1);
        assert_eq!(attrs[1].format, wgpu::VertexFormat::Float32x3);
    }

    #[test]
    fn vertex3d_texcoord_attribute() {
        let attrs = Vertex3D::layout().attributes;
        assert_eq!(attrs[2].offset, 24);
        assert_eq!(attrs[2].shader_location, 2);
        assert_eq!(attrs[2].format, wgpu::VertexFormat::Float32x2);
    }

    // ── bytemuck safety ──────────────────────────────────────────

    #[test]
    fn vertex3d_round_trips_through_bytes() {
        let v = Vertex3D {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [0.5, 0.5],
        };
        let bytes = bytemuck::bytes_of(&v);
        let back: &Vertex3D = bytemuck::from_bytes(bytes);
        assert_eq!(&v, back);
    }

    #[test]
    fn vertex3d_cast_slice_round_trips() {
        let verts = [
            Vertex3D {
                position: [0.0, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [0.0, 1.0],
            },
            Vertex3D {
                position: [-0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [0.0, 0.0],
            },
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&verts);
        assert_eq!(bytes.len(), 64); // 2 * 32
        let back: &[Vertex3D] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &verts);
    }

    // ── VertexBuffer layout delegation ───────────────────────────

    #[test]
    fn vertex_buffer_layout_delegates_to_vertex_type() {
        let from_type = Vertex3D::layout();
        let from_buf = VertexBuffer::<Vertex3D>::layout();
        assert_eq!(from_type.array_stride, from_buf.array_stride);
        assert_eq!(from_type.step_mode, from_buf.step_mode);
        assert_eq!(from_type.attributes.len(), from_buf.attributes.len());
    }

    // ── InstanceData layout ────────────────────────────────────────

    #[test]
    fn instance_data_size_is_64_bytes() {
        assert_eq!(mem::size_of::<InstanceData>(), 64);
    }

    #[test]
    fn instance_data_layout_stride_matches_size() {
        let layout = InstanceData::layout();
        assert_eq!(layout.array_stride, 64);
    }

    #[test]
    fn instance_data_layout_step_mode_is_instance() {
        let layout = InstanceData::layout();
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
    }

    #[test]
    fn instance_data_layout_has_four_attributes() {
        let layout = InstanceData::layout();
        assert_eq!(layout.attributes.len(), 4);
    }

    #[test]
    fn instance_data_attribute_locations_start_at_3() {
        let attrs = InstanceData::layout().attributes;
        for (i, attr) in attrs.iter().enumerate() {
            assert_eq!(attr.shader_location, 3 + i as u32);
            assert_eq!(attr.format, wgpu::VertexFormat::Float32x4);
            assert_eq!(attr.offset, (i * 16) as u64);
        }
    }

    #[test]
    fn instance_data_no_location_overlap_with_vertex3d() {
        let v_attrs = Vertex3D::layout().attributes;
        let i_attrs = InstanceData::layout().attributes;
        let v_locs: Vec<u32> = v_attrs.iter().map(|a| a.shader_location).collect();
        for attr in i_attrs {
            assert!(
                !v_locs.contains(&attr.shader_location),
                "location {} overlaps with Vertex3D",
                attr.shader_location
            );
        }
    }

    // ── InstanceData identity / from_mat4 ──────────────────────────

    #[test]
    fn instance_data_identity_is_identity_matrix() {
        let id = InstanceData::IDENTITY;
        for row in 0..4 {
            for col in 0..4 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert_eq!(id.model[row][col], expected);
            }
        }
    }

    #[test]
    fn instance_data_from_mat4_round_trips() {
        let mat = glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let inst = InstanceData::from_mat4(mat);
        let back = glam::Mat4::from_cols_array_2d(&inst.model);
        let a = mat.to_cols_array();
        let b = back.to_cols_array();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    // ── InstanceData bytemuck safety ────────────────────────────────

    #[test]
    fn instance_data_round_trips_through_bytes() {
        let inst = InstanceData::from_mat4(glam::Mat4::from_scale(glam::Vec3::splat(2.0)));
        let bytes = bytemuck::bytes_of(&inst);
        let back: &InstanceData = bytemuck::from_bytes(bytes);
        assert_eq!(&inst, back);
    }

    #[test]
    fn instance_data_cast_slice_round_trips() {
        let instances = [InstanceData::IDENTITY, InstanceData::from_mat4(
            glam::Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0)),
        )];
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        assert_eq!(bytes.len(), 128); // 2 * 64
        let back: &[InstanceData] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &instances);
    }
}
