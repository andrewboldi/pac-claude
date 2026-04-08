use crate::buffer::{InstanceData, Vertex, Vertex3D};
use crate::depth::{DepthBuffer, DEPTH_FORMAT};
use crate::material::Material;
use crate::GpuContext;

// ── Shader loading ──────────────────────────────────────────────────────

/// Load a WGSL shader module from source text.
pub fn load_shader(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

// ── Bind group helpers ──────────────────────────────────────────────────

/// Create a bind group layout with a single uniform buffer at binding 0.
pub fn uniform_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Create a bind group that binds a single buffer to binding 0 of the given layout.
pub fn uniform_bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}

// ── Pipeline descriptor ─────────────────────────────────────────────────

/// Configuration for creating a [`RenderPipeline`].
pub struct PipelineDescriptor<'a> {
    pub label: &'a str,
    pub shader: &'a wgpu::ShaderModule,
    pub vs_entry: &'a str,
    pub fs_entry: &'a str,
    pub vertex_layouts: &'a [wgpu::VertexBufferLayout<'a>],
    pub bind_group_layouts: &'a [&'a wgpu::BindGroupLayout],
    pub surface_format: wgpu::TextureFormat,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub cull_mode: Option<wgpu::Face>,
    pub topology: wgpu::PrimitiveTopology,
}

// ── RenderPipeline ──────────────────────────────────────────────────────

/// Wrapper around [`wgpu::RenderPipeline`] created from a [`PipelineDescriptor`].
pub struct RenderPipeline {
    inner: wgpu::RenderPipeline,
}

impl RenderPipeline {
    /// Create a render pipeline from the given descriptor.
    pub fn new(device: &wgpu::Device, desc: &PipelineDescriptor<'_>) -> Self {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", desc.label)),
            bind_group_layouts: desc.bind_group_layouts,
            push_constant_ranges: &[],
        });

        let depth_stencil = desc.depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let inner = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(desc.label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: desc.shader,
                entry_point: Some(desc.vs_entry),
                buffers: desc.vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: desc.shader,
                entry_point: Some(desc.fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: desc.surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: desc.topology,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: desc.cull_mode,
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self { inner }
    }

    /// Create a pipeline for a vertex type using sensible defaults.
    ///
    /// Uses triangle-list topology, back-face culling, no depth, and no bind groups.
    pub fn for_vertex<V: Vertex>(
        device: &wgpu::Device,
        label: &str,
        shader: &wgpu::ShaderModule,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self::new(
            device,
            &PipelineDescriptor {
                label,
                shader,
                vs_entry: "vs_main",
                fs_entry: "fs_main",
                vertex_layouts: &[V::layout()],
                bind_group_layouts: &[],
                surface_format,
                depth_format: None,
                cull_mode: Some(wgpu::Face::Back),
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
        )
    }

    /// The underlying `wgpu::RenderPipeline`.
    #[inline]
    pub fn inner(&self) -> &wgpu::RenderPipeline {
        &self.inner
    }
}

// ── TrianglePipeline (smoke-test pipeline) ──────────────────────────────

/// Render pipeline for drawing a single hardcoded colored triangle.
///
/// Used to validate the GPU pipeline: shader compilation, pipeline creation,
/// command encoding, and surface presentation. Built on top of [`RenderPipeline`].
pub struct TrianglePipeline {
    pipeline: RenderPipeline,
}

impl TrianglePipeline {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = load_shader(
            device,
            "basic.wgsl",
            include_str!("../../../assets/shaders/basic.wgsl"),
        );

        let pipeline = RenderPipeline::new(
            device,
            &PipelineDescriptor {
                label: "triangle_pipeline",
                shader: &shader,
                vs_entry: "vs_main",
                fs_entry: "fs_main",
                vertex_layouts: &[],
                bind_group_layouts: &[],
                surface_format,
                depth_format: Some(DEPTH_FORMAT),
                cull_mode: Some(wgpu::Face::Back),
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
        );

        Self { pipeline }
    }

    /// Acquire the surface texture, encode a render pass that draws the triangle,
    /// submit commands, and present.
    pub fn render_frame(
        &self,
        gpu: &GpuContext,
        depth: &DepthBuffer,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("triangle_encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("triangle_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth.view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(self.pipeline.inner());
            pass.draw(0..3, 0..1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// ── PhongPipeline ────────────────────────────────────────────────────────

/// Render pipeline for Phong-lit meshes with material textures.
///
/// Bind group layout:
///   - Group 0: Scene uniforms (view_proj, camera_pos)
///   - Group 1: Light uniforms (ambient, directional, point lights)
///   - Group 2: Material (uniforms + diffuse texture + sampler)
pub struct PhongPipeline {
    pipeline: RenderPipeline,
    scene_layout: wgpu::BindGroupLayout,
    light_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
}

impl PhongPipeline {
    /// Create a new Phong pipeline.
    ///
    /// The caller manages scene/light/material bind groups externally and
    /// passes them during rendering via [`draw`].
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = load_shader(
            device,
            "phong.wgsl",
            include_str!("../../../assets/shaders/phong.wgsl"),
        );

        let scene_layout = uniform_bind_group_layout(
            device,
            "phong_scene_layout",
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );
        let light_layout = uniform_bind_group_layout(
            device,
            "phong_light_layout",
            wgpu::ShaderStages::FRAGMENT,
        );
        let material_layout = Material::bind_group_layout(device);

        let pipeline = RenderPipeline::new(
            device,
            &PipelineDescriptor {
                label: "phong_pipeline",
                shader: &shader,
                vs_entry: "vs_main",
                fs_entry: "fs_main",
                vertex_layouts: &[Vertex3D::layout(), InstanceData::layout()],
                bind_group_layouts: &[&scene_layout, &light_layout, &material_layout],
                surface_format,
                depth_format: Some(DEPTH_FORMAT),
                cull_mode: Some(wgpu::Face::Back),
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
        );

        Self {
            pipeline,
            scene_layout,
            light_layout,
            material_layout,
        }
    }

    /// The underlying render pipeline for use with `set_pipeline`.
    #[inline]
    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline.inner()
    }

    /// Bind group layout for scene uniforms (group 0).
    #[inline]
    pub fn scene_layout(&self) -> &wgpu::BindGroupLayout {
        &self.scene_layout
    }

    /// Bind group layout for light uniforms (group 1).
    #[inline]
    pub fn light_layout(&self) -> &wgpu::BindGroupLayout {
        &self.light_layout
    }

    /// Bind group layout for material (group 2).
    #[inline]
    pub fn material_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_layout
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

    const MINIMAL_SHADER: &str = r#"
        @vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
        @fragment fn fs_main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    "#;

    const VERTEX3D_SHADER: &str = r#"
        struct VertexInput {
            @location(0) position: vec3f,
            @location(1) normal: vec3f,
            @location(2) texcoord: vec2f,
        }
        @vertex fn vs_main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 1.0);
        }
        @fragment fn fs_main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    "#;

    // ── load_shader ─────────────────────────────────────────────

    #[test]
    fn load_shader_creates_module() {
        let Some((device, _)) = test_device() else { return };
        let _shader = load_shader(&device, "test_shader", MINIMAL_SHADER);
    }

    // ── uniform_bind_group_layout ───────────────────────────────

    #[test]
    fn uniform_layout_vertex_visibility() {
        let Some((device, _)) = test_device() else { return };
        let _layout =
            uniform_bind_group_layout(&device, "test", wgpu::ShaderStages::VERTEX);
    }

    #[test]
    fn uniform_layout_fragment_visibility() {
        let Some((device, _)) = test_device() else { return };
        let _layout =
            uniform_bind_group_layout(&device, "test", wgpu::ShaderStages::FRAGMENT);
    }

    #[test]
    fn uniform_layout_vertex_fragment_visibility() {
        let Some((device, _)) = test_device() else { return };
        let _layout = uniform_bind_group_layout(
            &device,
            "test",
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );
    }

    // ── uniform_bind_group ──────────────────────────────────────

    #[test]
    fn uniform_bind_group_creates_group() {
        let Some((device, _)) = test_device() else { return };
        let layout =
            uniform_bind_group_layout(&device, "layout", wgpu::ShaderStages::VERTEX);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_buf"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let _bg = uniform_bind_group(&device, "bg", &layout, &buffer);
    }

    // ── RenderPipeline ──────────────────────────────────────────

    #[test]
    fn render_pipeline_no_depth_no_bindings() {
        let Some((device, _)) = test_device() else { return };
        let shader = load_shader(&device, "test", MINIMAL_SHADER);
        let _pipeline = RenderPipeline::new(
            &device,
            &PipelineDescriptor {
                label: "test_pipeline",
                shader: &shader,
                vs_entry: "vs_main",
                fs_entry: "fs_main",
                vertex_layouts: &[],
                bind_group_layouts: &[],
                surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
                depth_format: None,
                cull_mode: None,
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
        );
    }

    #[test]
    fn render_pipeline_with_depth() {
        let Some((device, _)) = test_device() else { return };
        let shader = load_shader(&device, "test", MINIMAL_SHADER);
        let _pipeline = RenderPipeline::new(
            &device,
            &PipelineDescriptor {
                label: "test_depth",
                shader: &shader,
                vs_entry: "vs_main",
                fs_entry: "fs_main",
                vertex_layouts: &[],
                bind_group_layouts: &[],
                surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
                depth_format: Some(DEPTH_FORMAT),
                cull_mode: Some(wgpu::Face::Back),
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
        );
    }

    #[test]
    fn render_pipeline_with_bind_group_layout() {
        let Some((device, _)) = test_device() else { return };
        let shader = load_shader(&device, "test", MINIMAL_SHADER);
        let layout =
            uniform_bind_group_layout(&device, "test", wgpu::ShaderStages::VERTEX);
        let _pipeline = RenderPipeline::new(
            &device,
            &PipelineDescriptor {
                label: "test_bg",
                shader: &shader,
                vs_entry: "vs_main",
                fs_entry: "fs_main",
                vertex_layouts: &[],
                bind_group_layouts: &[&layout],
                surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
                depth_format: None,
                cull_mode: None,
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
        );
    }

    #[test]
    fn render_pipeline_inner_returns_reference() {
        let Some((device, _)) = test_device() else { return };
        let shader = load_shader(&device, "test", MINIMAL_SHADER);
        let pipeline = RenderPipeline::new(
            &device,
            &PipelineDescriptor {
                label: "test_inner",
                shader: &shader,
                vs_entry: "vs_main",
                fs_entry: "fs_main",
                vertex_layouts: &[],
                bind_group_layouts: &[],
                surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
                depth_format: None,
                cull_mode: None,
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
        );
        let _inner = pipeline.inner();
    }

    #[test]
    fn render_pipeline_for_vertex() {
        let Some((device, _)) = test_device() else { return };
        let shader = load_shader(&device, "v3d", VERTEX3D_SHADER);
        let _pipeline = RenderPipeline::for_vertex::<Vertex3D>(
            &device,
            "test_for_vertex",
            &shader,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
    }

    #[test]
    fn render_pipeline_line_list_topology() {
        let Some((device, _)) = test_device() else { return };
        let shader = load_shader(&device, "test", MINIMAL_SHADER);
        let _pipeline = RenderPipeline::new(
            &device,
            &PipelineDescriptor {
                label: "test_lines",
                shader: &shader,
                vs_entry: "vs_main",
                fs_entry: "fs_main",
                vertex_layouts: &[],
                bind_group_layouts: &[],
                surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
                depth_format: None,
                cull_mode: None,
                topology: wgpu::PrimitiveTopology::LineList,
            },
        );
    }

    // ── TrianglePipeline ────────────────────────────────────────

    #[test]
    fn triangle_pipeline_creates() {
        let Some((device, _)) = test_device() else { return };
        let _tp = TrianglePipeline::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    // ── PhongPipeline ───────────────────────────────────────────

    #[test]
    fn phong_pipeline_creates() {
        let Some((device, _)) = test_device() else { return };
        let _pp = PhongPipeline::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn phong_pipeline_exposes_layouts() {
        let Some((device, _)) = test_device() else { return };
        let pp = PhongPipeline::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
        let _scene = pp.scene_layout();
        let _light = pp.light_layout();
        let _mat = pp.material_layout();
        let _inner = pp.pipeline();
    }
}
