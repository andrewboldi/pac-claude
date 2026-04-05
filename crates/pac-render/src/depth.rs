/// Depth format used for all depth buffers in the engine.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Manages a depth texture and its view for depth testing and 3D occlusion.
///
/// Create one per render target size. Recreate on resize via [`DepthBuffer::resize`].
pub struct DepthBuffer {
    view: wgpu::TextureView,
    _texture: wgpu::Texture,
}

impl DepthBuffer {
    /// Create a new depth buffer matching the given dimensions.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            view,
            _texture: texture,
        }
    }

    /// Recreate the depth buffer for new dimensions (e.g. after window resize).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        *self = Self::new(device, width, height);
    }

    /// The texture view to attach to a render pass depth/stencil attachment.
    #[inline]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The texture format of this depth buffer.
    #[inline]
    pub fn format(&self) -> wgpu::TextureFormat {
        DEPTH_FORMAT
    }
}
