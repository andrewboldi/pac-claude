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
}
