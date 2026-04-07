//! Render orchestrator: walks the scene graph, batches by material, submits draw calls.
//!
//! The [`Renderer`] owns scene-level GPU resources (uniform buffer, bind group)
//! and a pool of instance buffers reused across frames. Each frame it:
//!
//! 1. Updates scene uniforms (view-projection matrix, camera position)
//! 2. Collects renderable nodes from the [`SceneGraph`]
//! 3. Batches them by `(material, mesh)` to minimize bind-group switches
//! 4. Submits instanced draw calls through the [`PhongPipeline`]

use std::collections::HashMap;

use glam::Mat4;

use crate::buffer::{InstanceBuffer, InstanceData, UniformBuffer};
use crate::camera::Camera;
use crate::context::GpuContext;
use crate::depth::DepthBuffer;
use crate::light::LightManager;
use crate::material::Material;
use crate::mesh::GpuMesh;
use crate::pipeline::PhongPipeline;
use crate::scene::SceneGraph;

// ── SceneUniforms ──────────────────────────────────────────────────────

/// GPU-side scene uniforms matching the WGSL `SceneUniforms` struct.
///
/// Layout (80 bytes):
///
/// | Field        | Offset | Format       |
/// |--------------|--------|--------------|
/// | `view_proj`  | 0      | mat4x4<f32>  |
/// | `camera_pos` | 64     | vec4<f32>    |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

// ── DrawBatch (internal) ───────────────────────────────────────────────

/// A batched draw call: one mesh + one material + N instance transforms.
struct DrawBatch {
    mesh_index: usize,
    material_index: usize,
    instances: Vec<InstanceData>,
}

// ── Renderer ───────────────────────────────────────────────────────────

/// Render orchestrator that walks the scene graph, batches by material,
/// and submits instanced draw calls via the Phong pipeline.
///
/// Owns the scene uniform buffer/bind group and a pool of reusable
/// instance buffers that grow to fit the number of batches per frame.
pub struct Renderer {
    scene_buffer: UniformBuffer<SceneUniforms>,
    scene_bind_group: wgpu::BindGroup,
    instance_pool: Vec<InstanceBuffer>,
}

impl Renderer {
    /// Create a new renderer.
    ///
    /// Allocates the scene uniform buffer and bind group using the
    /// pipeline's scene layout (group 0).
    pub fn new(device: &wgpu::Device, pipeline: &PhongPipeline) -> Self {
        let uniforms = SceneUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
        };
        let scene_buffer = UniformBuffer::new(device, "scene_uniforms", &uniforms);
        let scene_bind_group = crate::pipeline::uniform_bind_group(
            device,
            "scene_bind_group",
            pipeline.scene_layout(),
            scene_buffer.buffer(),
        );

        Self {
            scene_buffer,
            scene_bind_group,
            instance_pool: Vec::new(),
        }
    }

    /// Render one frame.
    ///
    /// Acquires the surface texture, walks the scene graph to collect
    /// batched draw calls, and submits them through the Phong pipeline.
    ///
    /// `meshes` and `materials` are indexed by the values stored on
    /// [`SceneNode`](crate::scene::SceneNode). Nodes whose mesh or
    /// material index is out of range are silently skipped. Nodes with
    /// `material: None` default to material index 0.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        depth: &DepthBuffer,
        pipeline: &PhongPipeline,
        camera: &Camera,
        lights: &LightManager,
        scene: &SceneGraph,
        meshes: &[GpuMesh],
        materials: &[Material],
    ) -> Result<(), wgpu::SurfaceError> {
        // 1. Upload scene uniforms.
        let vp = camera.view_projection_matrix();
        let scene_uniforms = SceneUniforms {
            view_proj: vp.to_cols_array_2d(),
            camera_pos: [
                camera.position.x,
                camera.position.y,
                camera.position.z,
                1.0,
            ],
        };
        self.scene_buffer.write(&gpu.queue, &scene_uniforms);

        // 2. Upload light data.
        lights.write(&gpu.queue);

        // 3. Collect and batch draw calls, sorted by material.
        let batches = collect_batches(scene);

        // 4. Prepare instance buffers (must happen before the render pass).
        self.prepare_instance_buffers(&gpu.device, &gpu.queue, &batches);

        // 5. Acquire surface texture.
        let output = gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            gpu.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("renderer_encoder"),
                });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("phong_pass"),
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

            pass.set_pipeline(pipeline.pipeline());
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            pass.set_bind_group(1, lights.bind_group(), &[]);

            for (i, batch) in batches.iter().enumerate() {
                if batch.mesh_index >= meshes.len() || batch.material_index >= materials.len() {
                    continue;
                }

                pass.set_bind_group(2, materials[batch.material_index].bind_group(), &[]);
                meshes[batch.mesh_index].draw_instanced(&mut pass, &self.instance_pool[i]);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Grow the instance pool as needed and upload batch data.
    fn prepare_instance_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batches: &[DrawBatch],
    ) {
        // Grow pool to match batch count.
        while self.instance_pool.len() < batches.len() {
            self.instance_pool.push(InstanceBuffer::new(
                device,
                "batch_instances",
                &[InstanceData::IDENTITY],
            ));
        }

        // Upload instance data for each batch.
        for (i, batch) in batches.iter().enumerate() {
            self.instance_pool[i].write(device, queue, &batch.instances);
        }
    }
}

// ── Batching ───────────────────────────────────────────────────────────

/// Collect renderable nodes from the scene graph and batch by
/// `(material_index, mesh_index)`.
///
/// Returns batches sorted by material index first, then mesh index, to
/// minimize bind-group switches during rendering.
fn collect_batches(scene: &SceneGraph) -> Vec<DrawBatch> {
    let mut batch_map: HashMap<(usize, usize), Vec<InstanceData>> = HashMap::new();

    for (_handle, mesh_idx, material_opt, world_matrix) in scene.renderable_nodes() {
        let mat_idx = material_opt.unwrap_or(0);
        batch_map
            .entry((mat_idx, mesh_idx))
            .or_default()
            .push(InstanceData::from_mat4(world_matrix));
    }

    let mut batches: Vec<DrawBatch> = batch_map
        .into_iter()
        .map(|((material_index, mesh_index), instances)| DrawBatch {
            mesh_index,
            material_index,
            instances,
        })
        .collect();

    batches.sort_by_key(|b| (b.material_index, b.mesh_index));
    batches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneGraph;
    use glam::Vec3;
    use pac_math::Transform;

    // ── SceneUniforms layout ────────────────────────────────────────

    #[test]
    fn scene_uniforms_size_is_80_bytes() {
        assert_eq!(std::mem::size_of::<SceneUniforms>(), 80);
    }

    #[test]
    fn scene_uniforms_round_trips_through_bytes() {
        let u = SceneUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [1.0, 2.0, 3.0, 1.0],
        };
        let bytes = bytemuck::bytes_of(&u);
        let back: &SceneUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(back.camera_pos, u.camera_pos);
    }

    // ── collect_batches ─────────────────────────────────────────────

    #[test]
    fn empty_scene_produces_no_batches() {
        let scene = SceneGraph::new();
        let batches = collect_batches(&scene);
        assert!(batches.is_empty());
    }

    #[test]
    fn nodes_without_mesh_are_skipped() {
        let mut scene = SceneGraph::new();
        let _n = scene.add_child(scene.root(), Transform::IDENTITY);
        // No mesh assigned.
        scene.update_world_matrices();
        let batches = collect_batches(&scene);
        assert!(batches.is_empty());
    }

    #[test]
    fn single_node_produces_one_batch() {
        let mut scene = SceneGraph::new();
        let n = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(n).mesh = Some(0);
        scene.node_mut(n).material = Some(0);
        scene.update_world_matrices();

        let batches = collect_batches(&scene);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].mesh_index, 0);
        assert_eq!(batches[0].material_index, 0);
        assert_eq!(batches[0].instances.len(), 1);
    }

    #[test]
    fn same_mesh_and_material_batched_together() {
        let mut scene = SceneGraph::new();
        for _ in 0..5 {
            let n = scene.add_child(scene.root(), Transform::IDENTITY);
            scene.node_mut(n).mesh = Some(0);
            scene.node_mut(n).material = Some(0);
        }
        scene.update_world_matrices();

        let batches = collect_batches(&scene);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].instances.len(), 5);
    }

    #[test]
    fn different_materials_produce_separate_batches() {
        let mut scene = SceneGraph::new();
        for mat in 0..3 {
            let n = scene.add_child(scene.root(), Transform::IDENTITY);
            scene.node_mut(n).mesh = Some(0);
            scene.node_mut(n).material = Some(mat);
        }
        scene.update_world_matrices();

        let batches = collect_batches(&scene);
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn different_meshes_same_material_produce_separate_batches() {
        let mut scene = SceneGraph::new();
        for mesh in 0..3 {
            let n = scene.add_child(scene.root(), Transform::IDENTITY);
            scene.node_mut(n).mesh = Some(mesh);
            scene.node_mut(n).material = Some(0);
        }
        scene.update_world_matrices();

        let batches = collect_batches(&scene);
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn batches_sorted_by_material_then_mesh() {
        let mut scene = SceneGraph::new();
        // Add in reverse order: mat 2 mesh 1, mat 1 mesh 0, mat 0 mesh 2.
        let configs = [(2, 1), (1, 0), (0, 2)];
        for &(mat, mesh) in &configs {
            let n = scene.add_child(scene.root(), Transform::IDENTITY);
            scene.node_mut(n).mesh = Some(mesh);
            scene.node_mut(n).material = Some(mat);
        }
        scene.update_world_matrices();

        let batches = collect_batches(&scene);
        assert_eq!(batches.len(), 3);
        assert_eq!((batches[0].material_index, batches[0].mesh_index), (0, 2));
        assert_eq!((batches[1].material_index, batches[1].mesh_index), (1, 0));
        assert_eq!((batches[2].material_index, batches[2].mesh_index), (2, 1));
    }

    #[test]
    fn none_material_defaults_to_zero() {
        let mut scene = SceneGraph::new();
        let n = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(n).mesh = Some(0);
        // material left as None
        scene.update_world_matrices();

        let batches = collect_batches(&scene);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].material_index, 0);
    }

    #[test]
    fn world_matrix_propagated_to_instances() {
        let mut scene = SceneGraph::new();
        let n = scene.add_child(
            scene.root(),
            Transform::from_position(Vec3::new(5.0, 0.0, 0.0)),
        );
        scene.node_mut(n).mesh = Some(0);
        scene.node_mut(n).material = Some(0);
        scene.update_world_matrices();

        let batches = collect_batches(&scene);
        let inst = &batches[0].instances[0];
        // Column 3 (translation) should contain (5, 0, 0, 1).
        assert!((inst.model[3][0] - 5.0).abs() < 1e-5);
        assert!((inst.model[3][1] - 0.0).abs() < 1e-5);
        assert!((inst.model[3][2] - 0.0).abs() < 1e-5);
        assert!((inst.model[3][3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mixed_batching_scenario() {
        let mut scene = SceneGraph::new();
        // 3 nodes sharing (mesh 0, mat 0)
        for _ in 0..3 {
            let n = scene.add_child(scene.root(), Transform::IDENTITY);
            scene.node_mut(n).mesh = Some(0);
            scene.node_mut(n).material = Some(0);
        }
        // 2 nodes sharing (mesh 1, mat 0) — same material, different mesh
        for _ in 0..2 {
            let n = scene.add_child(scene.root(), Transform::IDENTITY);
            scene.node_mut(n).mesh = Some(1);
            scene.node_mut(n).material = Some(0);
        }
        // 1 node with (mesh 0, mat 1)
        let n = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(n).mesh = Some(0);
        scene.node_mut(n).material = Some(1);

        scene.update_world_matrices();

        let batches = collect_batches(&scene);
        assert_eq!(batches.len(), 3);

        // Sorted: (mat 0, mesh 0), (mat 0, mesh 1), (mat 1, mesh 0)
        assert_eq!(batches[0].material_index, 0);
        assert_eq!(batches[0].mesh_index, 0);
        assert_eq!(batches[0].instances.len(), 3);

        assert_eq!(batches[1].material_index, 0);
        assert_eq!(batches[1].mesh_index, 1);
        assert_eq!(batches[1].instances.len(), 2);

        assert_eq!(batches[2].material_index, 1);
        assert_eq!(batches[2].mesh_index, 0);
        assert_eq!(batches[2].instances.len(), 1);
    }
}
