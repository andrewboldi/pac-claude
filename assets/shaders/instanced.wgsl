// Instanced rendering shader with per-instance model transforms.
//
// Vertex slot 0: Vertex3D (position, normal, tex_coords) — locations 0–2
// Vertex slot 1: InstanceData (model matrix columns)     — locations 3–6
// Bind group 0, binding 0: view-projection uniform

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
};

struct InstanceInput {
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
};

struct Uniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(
        instance.model_0,
        instance.model_1,
        instance.model_2,
        instance.model_3,
    );

    let world_pos = model * vec4<f32>(vertex.position, 1.0);

    // Normal transform using upper-left 3×3 of model matrix.
    // Correct for uniform and non-uniform scale when the model matrix is
    // orthogonal; downstream tasks can pass a separate normal matrix if needed.
    let normal_mat = mat3x3<f32>(
        model[0].xyz,
        model[1].xyz,
        model[2].xyz,
    );

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * world_pos;
    out.world_normal = normalize(normal_mat * vertex.normal);
    out.tex_coords = vertex.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple directional light for visualisation.
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let n_dot_l = max(dot(in.world_normal, light_dir), 0.0);
    let ambient = 0.15;
    let brightness = ambient + n_dot_l * (1.0 - ambient);
    return vec4<f32>(vec3<f32>(brightness), 1.0);
}
