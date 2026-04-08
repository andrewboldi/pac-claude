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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: None,
                    ..Default::default()
                })
                .await?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()
        })
    }

    #[test]
    fn depth_format_is_depth32float() {
        assert_eq!(DEPTH_FORMAT, wgpu::TextureFormat::Depth32Float);
    }

    #[test]
    fn new_creates_buffer() {
        let Some((device, _)) = test_device() else { return };
        let db = DepthBuffer::new(&device, 800, 600);
        assert_eq!(db.format(), DEPTH_FORMAT);
    }

    #[test]
    fn format_returns_constant() {
        let Some((device, _)) = test_device() else { return };
        let db = DepthBuffer::new(&device, 64, 64);
        assert_eq!(db.format(), wgpu::TextureFormat::Depth32Float);
    }

    #[test]
    fn view_returns_reference() {
        let Some((device, _)) = test_device() else { return };
        let db = DepthBuffer::new(&device, 128, 128);
        let _view = db.view();
    }

    #[test]
    fn resize_replaces_buffer() {
        let Some((device, _)) = test_device() else { return };
        let mut db = DepthBuffer::new(&device, 100, 100);
        db.resize(&device, 200, 300);
        assert_eq!(db.format(), DEPTH_FORMAT);
    }

    #[test]
    fn new_with_small_dimensions() {
        let Some((device, _)) = test_device() else { return };
        let db = DepthBuffer::new(&device, 1, 1);
        assert_eq!(db.format(), DEPTH_FORMAT);
    }

    #[test]
    fn new_with_non_square_dimensions() {
        let Some((device, _)) = test_device() else { return };
        let db = DepthBuffer::new(&device, 1920, 1080);
        assert_eq!(db.format(), DEPTH_FORMAT);
    }
}
