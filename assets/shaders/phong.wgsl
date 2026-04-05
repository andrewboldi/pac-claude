// Phong lighting shader.
//
// Vertex slot 0: Vertex3D (position, normal, tex_coords) -- locations 0-2
// Vertex slot 1: InstanceData (model matrix columns)      -- locations 3-6
// Bind group 0, binding 0: SceneUniforms (view_proj + camera_pos)

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Light parameters (directional light from upper-right-front).
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let light_color = vec3<f32>(1.0, 1.0, 1.0);

    // Material parameters.
    let mat_ambient = vec3<f32>(0.1, 0.1, 0.1);
    let mat_diffuse = vec3<f32>(0.7, 0.7, 0.7);
    let mat_specular = vec3<f32>(0.5, 0.5, 0.5);
    let shininess = 32.0;

    let n = normalize(in.world_normal);

    // Ambient.
    let ambient = mat_ambient * light_color;

    // Diffuse (Lambert).
    let n_dot_l = max(dot(n, light_dir), 0.0);
    let diffuse = mat_diffuse * light_color * n_dot_l;

    // Specular (Blinn-Phong).
    let view_dir = normalize(scene.camera_pos.xyz - in.world_position);
    let half_dir = normalize(light_dir + view_dir);
    let spec_angle = max(dot(n, half_dir), 0.0);
    let specular = mat_specular * light_color * pow(spec_angle, shininess);

    let color = ambient + diffuse + specular;
    return vec4<f32>(color, 1.0);
}
