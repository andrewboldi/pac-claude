// Phong lighting shader.
//
// Vertex slot 0: Vertex3D (position, normal, tex_coords) -- locations 0-2
// Vertex slot 1: InstanceData (model matrix columns)      -- locations 3-6
// Bind group 0, binding 0: SceneUniforms (view_proj + camera_pos)
// Bind group 1, binding 0: LightUniforms (ambient, directional, point lights)

const MAX_POINT_LIGHTS: u32 = 4u;

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

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> scene: SceneUniforms;

// ── Light data ──────────────────────────────────────────────────────────

struct DirectionalLightData {
    direction: vec4<f32>,  // xyz = toward-light direction, w = 0
    color: vec4<f32>,      // xyz = RGB, w = intensity
};

struct PointLightData {
    position: vec4<f32>,     // xyz = world pos, w = 0
    color: vec4<f32>,        // xyz = RGB, w = intensity
    attenuation: vec4<f32>,  // x = constant, y = linear, z = quadratic, w = 0
};

struct LightUniforms {
    ambient: vec4<f32>,
    directional: DirectionalLightData,
    point_lights: array<PointLightData, 4>,
    num_point_lights: u32,
};

@group(1) @binding(0)
var<uniform> lights: LightUniforms;

// ── Vertex shader ───────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
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

    // Normal transform using upper-left 3x3 of model matrix.
    let normal_mat = mat3x3<f32>(
        model[0].xyz,
        model[1].xyz,
        model[2].xyz,
    );

    var out: VertexOutput;
    out.clip_position = scene.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal = normalize(normal_mat * vertex.normal);
    out.tex_coords = vertex.tex_coords;
    return out;
}

// ── Fragment shader ─────────────────────────────────────────────────────

// Material parameters (fixed for now; a future material system will make these dynamic).
const MAT_DIFFUSE: vec3<f32> = vec3<f32>(0.7, 0.7, 0.7);
const MAT_SPECULAR: vec3<f32> = vec3<f32>(0.5, 0.5, 0.5);
const SHININESS: f32 = 32.0;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let view_dir = normalize(scene.camera_pos.xyz - in.world_position);

    // Ambient contribution.
    var color = lights.ambient.xyz * MAT_DIFFUSE;

    // Directional light.
    let dir_l = normalize(lights.directional.direction.xyz);
    let dir_intensity = lights.directional.color.w;
    let dir_color = lights.directional.color.xyz * dir_intensity;

    let dir_ndl = max(dot(n, dir_l), 0.0);
    color += MAT_DIFFUSE * dir_color * dir_ndl;

    let dir_half = normalize(dir_l + view_dir);
    let dir_spec = pow(max(dot(n, dir_half), 0.0), SHININESS);
    color += MAT_SPECULAR * dir_color * dir_spec;

    // Point lights.
    for (var i = 0u; i < lights.num_point_lights; i++) {
        let pl = lights.point_lights[i];
        let to_light = pl.position.xyz - in.world_position;
        let dist = length(to_light);
        let pl_dir = to_light / dist;
        let pl_intensity = pl.color.w;
        let pl_color = pl.color.xyz * pl_intensity;

        // Attenuation.
        let att = 1.0 / (pl.attenuation.x + pl.attenuation.y * dist + pl.attenuation.z * dist * dist);

        // Diffuse.
        let pl_ndl = max(dot(n, pl_dir), 0.0);
        color += MAT_DIFFUSE * pl_color * pl_ndl * att;

        // Specular (Blinn-Phong).
        let pl_half = normalize(pl_dir + view_dir);
        let pl_spec = pow(max(dot(n, pl_half), 0.0), SHININESS);
        color += MAT_SPECULAR * pl_color * pl_spec * att;
    }

    return vec4<f32>(color, 1.0);
}
