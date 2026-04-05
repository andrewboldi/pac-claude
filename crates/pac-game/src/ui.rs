//! 2D UI overlay for score, lives, READY!, and GAME OVER text.
//!
//! Uses an embedded 8×8 bitmap font rendered as textured quads with an
//! orthographic projection. The overlay is drawn after the 3D scene with
//! alpha blending enabled and depth testing disabled.

use std::mem;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use pac_render::wgpu;
use wgpu::util::DeviceExt;

// ── Constants ──────────────────────────────────────────────────────────

const GLYPH_W: u32 = 8;
const GLYPH_H: u32 = 8;
const ATLAS_COLS: u32 = 16;
const ATLAS_ROWS: u32 = 6;
const ATLAS_W: u32 = ATLAS_COLS * GLYPH_W; // 128
const ATLAS_H: u32 = ATLAS_ROWS * GLYPH_H; // 48
const FONT_FIRST_CHAR: u8 = 32;

// ── Colors ─────────────────────────────────────────────────────────────

/// Pac-Man yellow.
pub const COLOR_YELLOW: [f32; 4] = [1.0, 0.878, 0.0, 1.0];
/// White for general text.
pub const COLOR_WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
/// Red for GAME OVER.
pub const COLOR_RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

const DEFAULT_SCALE: f32 = 16.0;
const LARGE_SCALE: f32 = 24.0;

// ── UiVertex ───────────────────────────────────────────────────────────

/// 2D vertex for UI overlay rendering.
///
/// | Field        | Offset | Format    |
/// |--------------|--------|-----------|
/// | `position`   | 0      | Float32x2 |
/// | `tex_coords` | 8      | Float32x2 |
/// | `color`      | 16     | Float32x4 |
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct UiVertex {
    pub position: [f32; 2],
    pub tex_coords: [f32; 2],
    pub color: [f32; 4],
}

const UI_VERTEX_ATTRS: [wgpu::VertexAttribute; 3] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x2,
    },
    wgpu::VertexAttribute {
        offset: 8,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x2,
    },
    wgpu::VertexAttribute {
        offset: 16,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32x4,
    },
];

impl UiVertex {
    /// Vertex buffer layout for the UI pipeline.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &UI_VERTEX_ATTRS,
        }
    }
}

// ── 8×8 bitmap font ───────────────────────────────────────────────────

/// 8×8 bitmap font covering ASCII 32–127 (96 glyphs).
/// Each glyph is 8 bytes (rows, top to bottom). Bit 7 = leftmost pixel.
#[rustfmt::skip]
const FONT_DATA: [[u8; 8]; 96] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // 32: Space
    [0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00], // 33: !
    [0x6C, 0x6C, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00], // 34: "
    [0x24, 0x24, 0x7E, 0x24, 0x7E, 0x24, 0x24, 0x00], // 35: #
    [0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00], // 36: $
    [0x00, 0x62, 0x64, 0x08, 0x10, 0x26, 0x46, 0x00], // 37: %
    [0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00], // 38: &
    [0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00], // 39: '
    [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00], // 40: (
    [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00], // 41: )
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00], // 42: *
    [0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00], // 43: +
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30], // 44: ,
    [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00], // 45: -
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00], // 46: .
    [0x02, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00], // 47: /
    [0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00], // 48: 0
    [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00], // 49: 1
    [0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00], // 50: 2
    [0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00], // 51: 3
    [0x0C, 0x1C, 0x2C, 0x4C, 0x7E, 0x0C, 0x0C, 0x00], // 52: 4
    [0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00], // 53: 5
    [0x3C, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x3C, 0x00], // 54: 6
    [0x7E, 0x06, 0x0C, 0x18, 0x18, 0x18, 0x18, 0x00], // 55: 7
    [0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00], // 56: 8
    [0x3C, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x3C, 0x00], // 57: 9
    [0x00, 0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00], // 58: :
    [0x00, 0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x30], // 59: ;
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00], // 60: <
    [0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00], // 61: =
    [0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60, 0x00], // 62: >
    [0x3C, 0x66, 0x06, 0x0C, 0x18, 0x00, 0x18, 0x00], // 63: ?
    [0x3C, 0x66, 0x6E, 0x6A, 0x6E, 0x60, 0x3C, 0x00], // 64: @
    [0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00], // 65: A
    [0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00], // 66: B
    [0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00], // 67: C
    [0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00], // 68: D
    [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00], // 69: E
    [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00], // 70: F
    [0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3C, 0x00], // 71: G
    [0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00], // 72: H
    [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00], // 73: I
    [0x06, 0x06, 0x06, 0x06, 0x06, 0x66, 0x3C, 0x00], // 74: J
    [0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00], // 75: K
    [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00], // 76: L
    [0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00], // 77: M
    [0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00], // 78: N
    [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00], // 79: O
    [0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00], // 80: P
    [0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00], // 81: Q
    [0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00], // 82: R
    [0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00], // 83: S
    [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00], // 84: T
    [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00], // 85: U
    [0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00], // 86: V
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00], // 87: W
    [0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00], // 88: X
    [0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00], // 89: Y
    [0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00], // 90: Z
    [0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00], // 91: [
    [0x40, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00], // 92: backslash
    [0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00], // 93: ]
    [0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00], // 94: ^
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00], // 95: _
    [0x18, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00], // 96: `
    [0x00, 0x00, 0x3C, 0x06, 0x3E, 0x66, 0x3E, 0x00], // 97: a
    [0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x00], // 98: b
    [0x00, 0x00, 0x3C, 0x66, 0x60, 0x66, 0x3C, 0x00], // 99: c
    [0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3E, 0x00], // 100: d
    [0x00, 0x00, 0x3C, 0x66, 0x7E, 0x60, 0x3C, 0x00], // 101: e
    [0x1C, 0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x00], // 102: f
    [0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x3C], // 103: g
    [0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00], // 104: h
    [0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00], // 105: i
    [0x06, 0x00, 0x0E, 0x06, 0x06, 0x06, 0x66, 0x3C], // 106: j
    [0x60, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0x00], // 107: k
    [0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00], // 108: l
    [0x00, 0x00, 0x76, 0x7F, 0x6B, 0x63, 0x63, 0x00], // 109: m
    [0x00, 0x00, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00], // 110: n
    [0x00, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x00], // 111: o
    [0x00, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60], // 112: p
    [0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x06], // 113: q
    [0x00, 0x00, 0x7C, 0x66, 0x60, 0x60, 0x60, 0x00], // 114: r
    [0x00, 0x00, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x00], // 115: s
    [0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x1C, 0x00], // 116: t
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x00], // 117: u
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00], // 118: v
    [0x00, 0x00, 0x63, 0x63, 0x6B, 0x7F, 0x36, 0x00], // 119: w
    [0x00, 0x00, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x00], // 120: x
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x3C], // 121: y
    [0x00, 0x00, 0x7E, 0x0C, 0x18, 0x30, 0x7E, 0x00], // 122: z
    [0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00], // 123: {
    [0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00], // 124: |
    [0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00], // 125: }
    [0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // 126: ~
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // 127: DEL
];

// ── Font texture generation ────────────────────────────────────────────

/// Generate RGBA8 pixel data for the 128×48 font atlas.
fn generate_font_atlas() -> Vec<u8> {
    let mut rgba = vec![0u8; (ATLAS_W * ATLAS_H * 4) as usize];
    for (idx, glyph) in FONT_DATA.iter().enumerate() {
        let col = (idx as u32) % ATLAS_COLS;
        let row = (idx as u32) / ATLAS_COLS;
        let base_x = col * GLYPH_W;
        let base_y = row * GLYPH_H;
        for (gy, &row_bits) in glyph.iter().enumerate() {
            for gx in 0..GLYPH_W {
                if row_bits & (0x80 >> gx) != 0 {
                    let px = base_x + gx;
                    let py = base_y + gy as u32;
                    let offset = ((py * ATLAS_W + px) * 4) as usize;
                    rgba[offset] = 255;
                    rgba[offset + 1] = 255;
                    rgba[offset + 2] = 255;
                    rgba[offset + 3] = 255;
                }
            }
        }
    }
    rgba
}

// ── UiOverlay ──────────────────────────────────────────────────────────

/// 2D UI overlay renderer for bitmap font text.
///
/// # Usage
///
/// ```ignore
/// // Each frame:
/// ui.begin_frame();
/// ui.draw_score(12345);
/// ui.draw_lives(3);
/// ui.draw_ready();        // or ui.draw_game_over();
/// ui.prepare(&device);
///
/// // In a render pass with LoadOp::Load (after 3D scene):
/// ui.render(&mut pass);
/// ```
pub struct UiOverlay {
    pipeline: wgpu::RenderPipeline,
    _font_texture: wgpu::Texture,
    _font_view: wgpu::TextureView,
    _font_sampler: wgpu::Sampler,
    font_bind_group: wgpu::BindGroup,
    projection_buffer: wgpu::Buffer,
    projection_bind_group: wgpu::BindGroup,
    vertices: Vec<UiVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    screen_width: f32,
    screen_height: f32,
}

impl UiOverlay {
    /// Create a new UI overlay renderer.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        screen_width: u32,
        screen_height: u32,
    ) -> Self {
        // ── Font texture ───────────────────────────────────────────
        let atlas_data = generate_font_atlas();
        let font_size = wgpu::Extent3d {
            width: ATLAS_W,
            height: ATLAS_H,
            depth_or_array_layers: 1,
        };
        let font_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ui_font_texture"),
            size: font_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &font_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * ATLAS_W),
                rows_per_image: Some(ATLAS_H),
            },
            font_size,
        );
        let font_view = font_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let font_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui_font_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ── Font bind group ────────────────────────────────────────
        let font_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui_font_bgl"),
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
        });
        let font_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui_font_bg"),
            layout: &font_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&font_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&font_sampler),
                },
            ],
        });

        // ── Projection uniform ─────────────────────────────────────
        let proj = ortho_matrix(screen_width as f32, screen_height as f32);
        let projection_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui_projection"),
            contents: bytemuck::cast_slice(&proj),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let proj_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui_proj_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let projection_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui_proj_bg"),
            layout: &proj_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: projection_buffer.as_entire_binding(),
            }],
        });

        // ── Pipeline (alpha blending, no depth) ────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/ui.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui_pipeline_layout"),
            bind_group_layouts: &[&proj_bgl, &font_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[UiVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            _font_texture: font_texture,
            _font_view: font_view,
            _font_sampler: font_sampler,
            font_bind_group,
            projection_buffer,
            projection_bind_group,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
            screen_width: screen_width as f32,
            screen_height: screen_height as f32,
        }
    }

    /// Update the projection matrix after a window resize.
    pub fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.screen_width = width as f32;
            self.screen_height = height as f32;
            let proj = ortho_matrix(self.screen_width, self.screen_height);
            queue.write_buffer(&self.projection_buffer, 0, bytemuck::cast_slice(&proj));
        }
    }

    /// Clear all pending vertices. Call once at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Draw a string at pixel position `(x, y)` with the given scale and color.
    ///
    /// `scale` sets the pixel size of each glyph (width and height are equal).
    /// Characters outside the printable ASCII range (32–127) are drawn as blanks.
    pub fn draw_text(&mut self, text: &str, x: f32, y: f32, scale: f32, color: [f32; 4]) {
        let mut cursor_x = x;
        for ch in text.chars() {
            let code = ch as u32;
            if code < FONT_FIRST_CHAR as u32 || code > 127 {
                cursor_x += scale;
                continue;
            }
            let idx = code - FONT_FIRST_CHAR as u32;
            let col = idx % ATLAS_COLS;
            let row = idx / ATLAS_COLS;

            let u0 = col as f32 / ATLAS_COLS as f32;
            let u1 = (col + 1) as f32 / ATLAS_COLS as f32;
            let v0 = row as f32 / ATLAS_ROWS as f32;
            let v1 = (row + 1) as f32 / ATLAS_ROWS as f32;

            let x0 = cursor_x;
            let y0 = y;
            let x1 = cursor_x + scale;
            let y1 = y + scale;

            let base = self.vertices.len() as u32;
            self.vertices.push(UiVertex {
                position: [x0, y0],
                tex_coords: [u0, v0],
                color,
            });
            self.vertices.push(UiVertex {
                position: [x1, y0],
                tex_coords: [u1, v0],
                color,
            });
            self.vertices.push(UiVertex {
                position: [x0, y1],
                tex_coords: [u0, v1],
                color,
            });
            self.vertices.push(UiVertex {
                position: [x1, y1],
                tex_coords: [u1, v1],
                color,
            });

            self.indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base + 2,
                base + 1,
                base + 3,
            ]);

            cursor_x += scale;
        }
    }

    /// Draw the score at the top-left of the screen.
    pub fn draw_score(&mut self, score: u32) {
        let text = format!("SCORE  {:>7}", score);
        self.draw_text(&text, 16.0, 8.0, DEFAULT_SCALE, COLOR_WHITE);
    }

    /// Draw the lives count at the bottom-left of the screen.
    pub fn draw_lives(&mut self, lives: u32) {
        let text = format!("LIVES: {}", lives);
        self.draw_text(&text, 16.0, self.screen_height - 24.0, DEFAULT_SCALE, COLOR_YELLOW);
    }

    /// Draw "READY!" centered on screen in yellow.
    pub fn draw_ready(&mut self) {
        let text = "READY!";
        let text_width = text.len() as f32 * LARGE_SCALE;
        let x = (self.screen_width - text_width) / 2.0;
        let y = self.screen_height / 2.0 - LARGE_SCALE / 2.0;
        self.draw_text(text, x, y, LARGE_SCALE, COLOR_YELLOW);
    }

    /// Draw "GAME OVER" centered on screen in red.
    pub fn draw_game_over(&mut self) {
        let text = "GAME OVER";
        let text_width = text.len() as f32 * LARGE_SCALE;
        let x = (self.screen_width - text_width) / 2.0;
        let y = self.screen_height / 2.0 - LARGE_SCALE / 2.0;
        self.draw_text(text, x, y, LARGE_SCALE, COLOR_RED);
    }

    /// Upload accumulated vertices and indices to the GPU.
    ///
    /// Call after all `draw_*` methods and before [`render`](Self::render).
    pub fn prepare(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            self.index_count = 0;
            self.vertex_buffer = None;
            self.index_buffer = None;
            return;
        }
        self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui_vertices"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        self.index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui_indices"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
        self.index_count = self.indices.len() as u32;
    }

    /// Record draw commands into a render pass.
    ///
    /// The render pass should use `LoadOp::Load` for the color attachment (to
    /// preserve the 3D scene) and have no depth attachment.
    pub fn render<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>) {
        if self.index_count == 0 {
            return;
        }
        let vb = match self.vertex_buffer.as_ref() {
            Some(b) => b,
            None => return,
        };
        let ib = match self.index_buffer.as_ref() {
            Some(b) => b,
            None => return,
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.projection_bind_group, &[]);
        pass.set_bind_group(1, &self.font_bind_group, &[]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    /// Current screen dimensions in pixels.
    pub fn screen_size(&self) -> (f32, f32) {
        (self.screen_width, self.screen_height)
    }
}

/// Build an orthographic projection matrix mapping screen pixels to clip space.
/// Origin is top-left, Y points down.
fn ortho_matrix(width: f32, height: f32) -> [f32; 16] {
    Mat4::orthographic_rh(0.0, width, height, 0.0, -1.0, 1.0).to_cols_array()
}

/// Compute UV coordinates for a character in the font atlas.
/// Returns `(u0, v0, u1, v1)` or `None` if the character is out of range.
pub fn glyph_uvs(ch: char) -> Option<(f32, f32, f32, f32)> {
    let code = ch as u32;
    if code < FONT_FIRST_CHAR as u32 || code > 127 {
        return None;
    }
    let idx = code - FONT_FIRST_CHAR as u32;
    let col = idx % ATLAS_COLS;
    let row = idx / ATLAS_COLS;
    Some((
        col as f32 / ATLAS_COLS as f32,
        row as f32 / ATLAS_ROWS as f32,
        (col + 1) as f32 / ATLAS_COLS as f32,
        (row + 1) as f32 / ATLAS_ROWS as f32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── UiVertex layout ─────────────────────────────────────────────

    #[test]
    fn ui_vertex_size_is_32_bytes() {
        assert_eq!(mem::size_of::<UiVertex>(), 32);
    }

    #[test]
    fn ui_vertex_layout_stride_matches_size() {
        let layout = UiVertex::layout();
        assert_eq!(layout.array_stride, 32);
    }

    #[test]
    fn ui_vertex_layout_step_mode_is_vertex() {
        let layout = UiVertex::layout();
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
    }

    #[test]
    fn ui_vertex_layout_has_three_attributes() {
        let layout = UiVertex::layout();
        assert_eq!(layout.attributes.len(), 3);
    }

    #[test]
    fn ui_vertex_position_attribute() {
        let attrs = UiVertex::layout().attributes;
        assert_eq!(attrs[0].offset, 0);
        assert_eq!(attrs[0].shader_location, 0);
        assert_eq!(attrs[0].format, wgpu::VertexFormat::Float32x2);
    }

    #[test]
    fn ui_vertex_texcoord_attribute() {
        let attrs = UiVertex::layout().attributes;
        assert_eq!(attrs[1].offset, 8);
        assert_eq!(attrs[1].shader_location, 1);
        assert_eq!(attrs[1].format, wgpu::VertexFormat::Float32x2);
    }

    #[test]
    fn ui_vertex_color_attribute() {
        let attrs = UiVertex::layout().attributes;
        assert_eq!(attrs[2].offset, 16);
        assert_eq!(attrs[2].shader_location, 2);
        assert_eq!(attrs[2].format, wgpu::VertexFormat::Float32x4);
    }

    // ── bytemuck safety ─────────────────────────────────────────────

    #[test]
    fn ui_vertex_round_trips_through_bytes() {
        let v = UiVertex {
            position: [10.0, 20.0],
            tex_coords: [0.5, 0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        };
        let bytes = bytemuck::bytes_of(&v);
        let back: &UiVertex = bytemuck::from_bytes(bytes);
        assert_eq!(&v, back);
    }

    #[test]
    fn ui_vertex_cast_slice_round_trips() {
        let verts = [
            UiVertex {
                position: [0.0, 0.0],
                tex_coords: [0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            UiVertex {
                position: [100.0, 50.0],
                tex_coords: [1.0, 1.0],
                color: [0.0, 0.0, 0.0, 0.5],
            },
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&verts);
        assert_eq!(bytes.len(), 64);
        let back: &[UiVertex] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &verts);
    }

    // ── Font data ───────────────────────────────────────────────────

    #[test]
    fn font_data_has_96_glyphs() {
        assert_eq!(FONT_DATA.len(), 96);
    }

    #[test]
    fn font_space_is_blank() {
        assert_eq!(FONT_DATA[0], [0; 8]);
    }

    #[test]
    fn font_exclamation_is_not_blank() {
        assert_ne!(FONT_DATA[1], [0; 8]);
    }

    #[test]
    fn font_digits_are_not_blank() {
        for i in 0..10u8 {
            let idx = (b'0' + i - FONT_FIRST_CHAR) as usize;
            assert_ne!(FONT_DATA[idx], [0; 8], "digit {} should not be blank", i);
        }
    }

    #[test]
    fn font_uppercase_letters_are_not_blank() {
        for c in b'A'..=b'Z' {
            let idx = (c - FONT_FIRST_CHAR) as usize;
            assert_ne!(FONT_DATA[idx], [0; 8], "'{}' should not be blank", c as char);
        }
    }

    // ── Font atlas generation ───────────────────────────────────────

    #[test]
    fn atlas_has_correct_byte_count() {
        let atlas = generate_font_atlas();
        assert_eq!(atlas.len(), (ATLAS_W * ATLAS_H * 4) as usize);
    }

    #[test]
    fn atlas_space_region_is_transparent() {
        let atlas = generate_font_atlas();
        // Space is at col=0, row=0 → pixels (0,0)..(8,8)
        for y in 0..GLYPH_H {
            for x in 0..GLYPH_W {
                let offset = ((y * ATLAS_W + x) * 4) as usize;
                assert_eq!(atlas[offset + 3], 0, "space pixel ({x},{y}) should be transparent");
            }
        }
    }

    #[test]
    fn atlas_exclamation_has_visible_pixels() {
        let atlas = generate_font_atlas();
        // '!' is at index 1 → col=1, row=0 → pixels (8,0)..(16,8)
        let mut has_pixel = false;
        for y in 0..GLYPH_H {
            for x in 0..GLYPH_W {
                let px = GLYPH_W + x; // col 1
                let offset = ((y * ATLAS_W + px) * 4) as usize;
                if atlas[offset + 3] == 255 {
                    has_pixel = true;
                }
            }
        }
        assert!(has_pixel, "exclamation mark should have visible pixels");
    }

    // ── UV calculation ──────────────────────────────────────────────

    #[test]
    fn glyph_uvs_space() {
        let (u0, v0, u1, v1) = glyph_uvs(' ').unwrap();
        assert!((u0 - 0.0).abs() < 1e-6);
        assert!((v0 - 0.0).abs() < 1e-6);
        assert!((u1 - 1.0 / 16.0).abs() < 1e-6);
        assert!((v1 - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn glyph_uvs_a() {
        // 'A' = 65, index = 33, col = 33 % 16 = 1, row = 33 / 16 = 2
        let (u0, v0, u1, v1) = glyph_uvs('A').unwrap();
        assert!((u0 - 1.0 / 16.0).abs() < 1e-6);
        assert!((v0 - 2.0 / 6.0).abs() < 1e-6);
        assert!((u1 - 2.0 / 16.0).abs() < 1e-6);
        assert!((v1 - 3.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn glyph_uvs_out_of_range_returns_none() {
        assert!(glyph_uvs('\x00').is_none());
        assert!(glyph_uvs('\x1F').is_none());
        assert!(glyph_uvs('\u{0080}').is_none());
    }

    #[test]
    fn glyph_uvs_tilde_is_last_visible() {
        assert!(glyph_uvs('~').is_some());
    }

    // ── Text vertex generation ──────────────────────────────────────

    /// Helper: create a minimal overlay-like state to test draw_text vertex generation.
    fn make_draw_state() -> (Vec<UiVertex>, Vec<u32>) {
        (Vec::new(), Vec::new())
    }

    fn draw_text_into(
        vertices: &mut Vec<UiVertex>,
        indices: &mut Vec<u32>,
        text: &str,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
    ) {
        let mut cursor_x = x;
        for ch in text.chars() {
            let code = ch as u32;
            if code < FONT_FIRST_CHAR as u32 || code > 127 {
                cursor_x += scale;
                continue;
            }
            let idx = code - FONT_FIRST_CHAR as u32;
            let col = idx % ATLAS_COLS;
            let row = idx / ATLAS_COLS;
            let u0 = col as f32 / ATLAS_COLS as f32;
            let u1 = (col + 1) as f32 / ATLAS_COLS as f32;
            let v0 = row as f32 / ATLAS_ROWS as f32;
            let v1 = (row + 1) as f32 / ATLAS_ROWS as f32;
            let x0 = cursor_x;
            let y0 = y;
            let x1 = cursor_x + scale;
            let y1 = y + scale;
            let base = vertices.len() as u32;
            vertices.push(UiVertex { position: [x0, y0], tex_coords: [u0, v0], color });
            vertices.push(UiVertex { position: [x1, y0], tex_coords: [u1, v0], color });
            vertices.push(UiVertex { position: [x0, y1], tex_coords: [u0, v1], color });
            vertices.push(UiVertex { position: [x1, y1], tex_coords: [u1, v1], color });
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
            cursor_x += scale;
        }
    }

    #[test]
    fn draw_text_empty_string_produces_no_vertices() {
        let (mut v, mut i) = make_draw_state();
        draw_text_into(&mut v, &mut i, "", 0.0, 0.0, 16.0, COLOR_WHITE);
        assert!(v.is_empty());
        assert!(i.is_empty());
    }

    #[test]
    fn draw_text_single_char_produces_4_vertices_6_indices() {
        let (mut v, mut i) = make_draw_state();
        draw_text_into(&mut v, &mut i, "A", 0.0, 0.0, 16.0, COLOR_WHITE);
        assert_eq!(v.len(), 4);
        assert_eq!(i.len(), 6);
    }

    #[test]
    fn draw_text_hello_produces_correct_counts() {
        let (mut v, mut i) = make_draw_state();
        draw_text_into(&mut v, &mut i, "HELLO", 0.0, 0.0, 16.0, COLOR_WHITE);
        assert_eq!(v.len(), 5 * 4);
        assert_eq!(i.len(), 5 * 6);
    }

    #[test]
    fn draw_text_positions_advance_by_scale() {
        let (mut v, mut i) = make_draw_state();
        draw_text_into(&mut v, &mut i, "AB", 10.0, 20.0, 16.0, COLOR_WHITE);
        // First char: x0=10
        assert!((v[0].position[0] - 10.0).abs() < 1e-6);
        // Second char: x0=26 (10+16)
        assert!((v[4].position[0] - 26.0).abs() < 1e-6);
    }

    #[test]
    fn draw_text_y_span_equals_scale() {
        let (mut v, mut i) = make_draw_state();
        draw_text_into(&mut v, &mut i, "X", 0.0, 100.0, 24.0, COLOR_WHITE);
        let y_top = v[0].position[1];
        let y_bot = v[2].position[1];
        assert!((y_top - 100.0).abs() < 1e-6);
        assert!((y_bot - 124.0).abs() < 1e-6);
    }

    #[test]
    fn draw_text_color_propagated() {
        let (mut v, mut i) = make_draw_state();
        draw_text_into(&mut v, &mut i, "R", 0.0, 0.0, 16.0, COLOR_RED);
        for vert in &v {
            assert_eq!(vert.color, COLOR_RED);
        }
    }

    // ── Orthographic projection ─────────────────────────────────────

    #[test]
    fn ortho_maps_origin_to_top_left() {
        let proj = Mat4::from_cols_array(&ortho_matrix(800.0, 600.0));
        let clip = proj * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        // Top-left in NDC should be (-1, 1)
        assert!((clip.x - (-1.0)).abs() < 1e-5);
        assert!((clip.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ortho_maps_bottom_right_correctly() {
        let proj = Mat4::from_cols_array(&ortho_matrix(800.0, 600.0));
        let clip = proj * glam::Vec4::new(800.0, 600.0, 0.0, 1.0);
        // Bottom-right in NDC should be (1, -1)
        assert!((clip.x - 1.0).abs() < 1e-5);
        assert!((clip.y - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn ortho_maps_center_to_origin() {
        let proj = Mat4::from_cols_array(&ortho_matrix(800.0, 600.0));
        let clip = proj * glam::Vec4::new(400.0, 300.0, 0.0, 1.0);
        assert!((clip.x).abs() < 1e-5);
        assert!((clip.y).abs() < 1e-5);
    }
}
