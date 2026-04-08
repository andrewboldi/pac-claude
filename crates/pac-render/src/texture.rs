//! GPU texture abstraction wrapping `wgpu::Texture`, `TextureView`, and `Sampler`.

use std::path::Path;

/// GPU texture with associated view and sampler, ready for binding in shaders.
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    /// Load a texture from PNG file bytes.
    ///
    /// Decodes the PNG into RGBA8, uploads it to the GPU, and creates a
    /// texture view and sampler with linear filtering.
    pub fn from_png_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
    ) -> Self {
        let img = image::load_from_memory(bytes)
            .expect("failed to decode PNG")
            .into_rgba8();
        let (width, height) = img.dimensions();
        let rgba = img.into_raw();

        Self::from_rgba8(device, queue, &rgba, width, height, label)
    }

    /// Load a texture from a PNG file on disk.
    pub fn from_png_path(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &Path,
        label: &str,
    ) -> Self {
        let bytes = std::fs::read(path).expect("failed to read texture file");
        Self::from_png_bytes(device, queue, &bytes, label)
    }

    /// Create a texture from raw RGBA8 pixel data.
    pub fn from_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: &[u8],
        width: u32,
        height: u32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{label}_sampler")),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }

    /// Bind group layout for a texture + sampler pair at bindings 0 and 1.
    pub fn bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    /// Create a bind group binding this texture's view and sampler.
    pub fn bind_group(
        &self,
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Texture dimensions (width, height).
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        let s = self.texture.size();
        (s.width, s.height)
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
    fn from_rgba8_1x1() {
        let Some((device, queue)) = test_device() else { return };
        let rgba = [255u8, 0, 0, 255];
        let tex = Texture::from_rgba8(&device, &queue, &rgba, 1, 1, "test_1x1");
        assert_eq!(tex.size(), (1, 1));
    }

    #[test]
    fn from_rgba8_16x16() {
        let Some((device, queue)) = test_device() else { return };
        let rgba = vec![128u8; 4 * 16 * 16];
        let tex = Texture::from_rgba8(&device, &queue, &rgba, 16, 16, "test_16x16");
        assert_eq!(tex.size(), (16, 16));
    }

    #[test]
    fn from_rgba8_non_square() {
        let Some((device, queue)) = test_device() else { return };
        let rgba = vec![0u8; 4 * 32 * 64];
        let tex = Texture::from_rgba8(&device, &queue, &rgba, 32, 64, "test_non_square");
        assert_eq!(tex.size(), (32, 64));
    }

    #[test]
    fn from_png_bytes_decodes() {
        let Some((device, queue)) = test_device() else { return };
        // Create a minimal 2x1 PNG in memory
        let mut buf = Vec::new();
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut buf);
            image::ImageEncoder::write_image(
                encoder,
                &[255, 0, 0, 255, 0, 255, 0, 255],
                2,
                1,
                image::ExtendedColorType::Rgba8,
            )
            .unwrap();
        }
        let tex = Texture::from_png_bytes(&device, &queue, &buf, "test_png");
        assert_eq!(tex.size(), (2, 1));
    }

    #[test]
    fn bind_group_layout_creates_layout() {
        let Some((device, _)) = test_device() else { return };
        let _layout = Texture::bind_group_layout(&device, "test_layout");
    }

    #[test]
    fn bind_group_creates_group() {
        let Some((device, queue)) = test_device() else { return };
        let layout = Texture::bind_group_layout(&device, "test_layout");
        let rgba = [255u8, 255, 255, 255];
        let tex = Texture::from_rgba8(&device, &queue, &rgba, 1, 1, "test");
        let _bg = tex.bind_group(&device, "test_bg", &layout);
    }

    #[test]
    fn size_matches_input_dimensions() {
        let Some((device, queue)) = test_device() else { return };
        for (w, h) in [(1, 1), (4, 4), (100, 50), (256, 256)] {
            let rgba = vec![0u8; 4 * w as usize * h as usize];
            let tex = Texture::from_rgba8(&device, &queue, &rgba, w, h, "test_size");
            assert_eq!(tex.size(), (w, h), "mismatch for {w}x{h}");
        }
    }

    #[test]
    fn texture_format_is_rgba8_srgb() {
        let Some((device, queue)) = test_device() else { return };
        let rgba = [0u8; 4];
        let tex = Texture::from_rgba8(&device, &queue, &rgba, 1, 1, "test_format");
        assert_eq!(tex.texture.format(), wgpu::TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn texture_usage_includes_binding_and_copy_dst() {
        let Some((device, queue)) = test_device() else { return };
        let rgba = [0u8; 4];
        let tex = Texture::from_rgba8(&device, &queue, &rgba, 1, 1, "test_usage");
        let usage = tex.texture.usage();
        assert!(usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert!(usage.contains(wgpu::TextureUsages::COPY_DST));
    }
}
