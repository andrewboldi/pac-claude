// 2D UI overlay shader
// Renders textured quads with per-vertex color and alpha blending.
// Used for score, lives, READY!, and GAME OVER text.

struct UiUniforms {
    projection: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: UiUniforms;

@group(1) @binding(0)
var font_texture: texture_2d<f32>;
@group(1) @binding(1)
var font_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.projection * vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(font_texture, font_sampler, in.tex_coords);
    // Font texture is white-on-transparent. Use red channel as alpha mask.
    let alpha = tex_color.r * in.color.a;
    if (alpha < 0.01) {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
