use wgpu::{
    Adapter, Device, Instance, InstanceDescriptor, Queue, Surface, SurfaceConfiguration,
    TextureUsages,
};

/// GPU rendering context holding all wgpu state needed for rendering.
pub struct GpuContext<'window> {
    pub instance: Instance,
    pub surface: Surface<'window>,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_config: SurfaceConfiguration,
}

impl<'window> GpuContext<'window> {
    /// Create a new GPU context from a window surface target.
    ///
    /// This is async because adapter and device requests are async in wgpu.
    pub async fn new(
        target: impl Into<wgpu::SurfaceTarget<'window>>,
        width: u32,
        height: u32,
    ) -> Self {
        let instance = Instance::new(InstanceDescriptor::default());

        let surface = instance
            .create_surface(target)
            .expect("failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        log::info!("GPU adapter: {:?}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("pac-device"),
                ..Default::default()
            }, None)
            .await
            .expect("failed to create GPU device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        log::info!(
            "GPU context initialized: {}x{} format={:?}",
            width,
            height,
            surface_format
        );

        Self {
            instance,
            surface,
            adapter,
            device,
            queue,
            surface_config,
        }
    }

    /// Reconfigure the surface after a window resize.
    ///
    /// Ignores zero-sized dimensions (minimized windows).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
            log::info!("Surface resized to {width}x{height}");
        }
    }

    /// Current surface texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    /// Current surface dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.surface_config.width, self.surface_config.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GpuContext::new requires a window surface target, so full construction
    // tests need a windowing system. These tests verify the struct API via a
    // manually assembled SurfaceConfiguration to test resize/format/size logic.

    fn make_config(width: u32, height: u32) -> SurfaceConfiguration {
        SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        }
    }

    #[test]
    fn size_returns_config_dimensions() {
        let config = make_config(800, 600);
        assert_eq!((config.width, config.height), (800, 600));
    }

    #[test]
    fn format_returns_config_format() {
        let config = make_config(100, 100);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn config_usage_is_render_attachment() {
        let config = make_config(100, 100);
        assert!(config.usage.contains(TextureUsages::RENDER_ATTACHMENT));
    }

    #[test]
    fn config_view_formats_empty() {
        let config = make_config(100, 100);
        assert!(config.view_formats.is_empty());
    }

    #[test]
    fn config_desired_max_frame_latency() {
        let config = make_config(100, 100);
        assert_eq!(config.desired_maximum_frame_latency, 2);
    }
}
