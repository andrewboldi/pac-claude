use crate::buffer::Vertex;
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
                depth_format: None,
                cull_mode: Some(wgpu::Face::Back),
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
        );

        Self { pipeline }
    }

    /// Acquire the surface texture, encode a render pass that draws the triangle,
    /// submit commands, and present.
    pub fn render_frame(&self, gpu: &GpuContext) -> Result<(), wgpu::SurfaceError> {
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
                depth_stencil_attachment: None,
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
