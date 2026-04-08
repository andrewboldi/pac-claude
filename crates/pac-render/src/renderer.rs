//! Render orchestrator: walks the scene graph, batches by material, submits draw calls.
//!
//! The [`Renderer`] owns the Phong pipeline, scene uniform buffer, and a
//! reusable instance buffer. Each frame it:
//!
//! 1. Updates per-frame scene uniforms (view-projection matrix, camera position).
//! 2. Collects renderable nodes from the [`SceneGraph`].
//! 3. Sorts them by `(material_index, mesh_index)` to minimise state changes.
//! 4. Issues one instanced draw call per unique (material, mesh) pair.

use glam::Mat4;

use crate::buffer::{InstanceBuffer, InstanceData, UniformBuffer};
use crate::camera::Camera;
use crate::context::GpuContext;
use crate::depth::DepthBuffer;
use crate::light::LightManager;
use crate::material::Material;
use crate::mesh::GpuMesh;
use crate::pipeline::{uniform_bind_group, PhongPipeline};
use crate::scene::SceneGraph;

// ── SceneUniforms ──────────────────────────────────────────────────────

/// GPU-side scene uniform data matching the WGSL `SceneUniforms` struct.
///
/// Layout (80 bytes):
/// - `view_proj`: 64 bytes (mat4x4<f32>)
/// - `camera_pos`: 16 bytes (vec4<f32>, xyz used, w padding)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

impl SceneUniforms {
    /// Build scene uniforms from a camera.
    pub fn from_camera(camera: &Camera) -> Self {
        let vp = camera.view_projection_matrix();
        Self {
            view_proj: vp.to_cols_array_2d(),
            camera_pos: [
                camera.position.x,
                camera.position.y,
                camera.position.z,
                0.0,
            ],
        }
    }
}

// ── DrawBatch (internal) ───────────────────────────────────────────────

/// A batch of instances sharing the same material and mesh.
struct DrawBatch {
    material_index: usize,
    mesh_index: usize,
    instances: Vec<InstanceData>,
}

// ── Renderer ───────────────────────────────────────────────────────────

/// Render orchestrator that walks the scene graph and issues batched draw calls.
///
/// Create once at startup. Call [`Renderer::render_frame`] each frame.
pub struct Renderer {
    phong: PhongPipeline,
    scene_buffer: UniformBuffer<SceneUniforms>,
    scene_bind_group: wgpu::BindGroup,
    instance_buffer: InstanceBuffer,
}

impl Renderer {
    /// Create a new renderer.
    ///
    /// `lights` must be the same [`LightManager`] used during rendering so the
    /// bind group layouts are compatible.
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let phong = PhongPipeline::new(device, surface_format);

        let scene_uniforms = SceneUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
        };
        let scene_buffer = UniformBuffer::new(device, "scene_uniforms", &scene_uniforms);
        let scene_bind_group = uniform_bind_group(
            device,
            "scene_bind_group",
            phong.scene_layout(),
            scene_buffer.buffer(),
        );

        let instance_buffer =
            InstanceBuffer::new(device, "renderer_instances", &[InstanceData::IDENTITY]);

        Self {
            phong,
            scene_buffer,
            scene_bind_group,
            instance_buffer,
        }
    }

    /// The material bind group layout, needed when creating [`Material`]s.
    #[inline]
    pub fn material_layout(&self) -> &wgpu::BindGroupLayout {
        self.phong.material_layout()
    }

    /// Render a frame: walk the scene graph, batch by material, and submit.
    ///
    /// # Arguments
    /// - `gpu` - The GPU context (device, queue, surface).
    /// - `depth` - The depth buffer for the current surface size.
    /// - `camera` - Current camera state.
    /// - `lights` - Scene lights (must have been written to the GPU this frame).
    /// - `scene` - The scene graph (world matrices must already be updated).
    /// - `meshes` - GPU meshes indexed by `SceneNode::mesh`.
    /// - `materials` - Materials indexed by `SceneNode::material`.
    /// - `default_material` - Fallback material for nodes with no material set.
    pub fn render_frame(
        &mut self,
        gpu: &GpuContext,
        depth: &DepthBuffer,
        camera: &Camera,
        lights: &LightManager,
        scene: &SceneGraph,
        meshes: &[GpuMesh],
        materials: &[Material],
        default_material: &Material,
    ) -> Result<(), wgpu::SurfaceError> {
        // 1. Upload scene uniforms.
        let scene_uniforms = SceneUniforms::from_camera(camera);
        self.scene_buffer.write(&gpu.queue, &scene_uniforms);

        // 2. Build batches and upload all instance data before the render pass.
        //    We concatenate all instances into one buffer and record per-batch
        //    ranges to avoid borrow conflicts during the pass.
        let batches = self.collect_batches(scene);
        let mut all_instances = Vec::new();
        let mut batch_ranges: Vec<(usize, usize, u32, u32)> = Vec::new(); // (mat, mesh, start, count)
        for batch in &batches {
            let start = all_instances.len() as u32;
            all_instances.extend_from_slice(&batch.instances);
            let count = batch.instances.len() as u32;
            batch_ranges.push((batch.material_index, batch.mesh_index, start, count));
        }
        if !all_instances.is_empty() {
            self.instance_buffer
                .write(&gpu.device, &gpu.queue, &all_instances);
        }

        // 3. Acquire surface texture.
        let output = gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // 4. Encode render pass.
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

            pass.set_pipeline(self.phong.pipeline());
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            pass.set_bind_group(1, lights.bind_group(), &[]);

            // Bind the shared instance buffer once (slot 1).
            pass.set_vertex_buffer(1, self.instance_buffer.slice());

            for &(mat_idx, mesh_idx, inst_start, inst_count) in &batch_ranges {
                let mat = materials.get(mat_idx).unwrap_or(default_material);
                pass.set_bind_group(2, mat.bind_group(), &[]);

                if let Some(mesh) = meshes.get(mesh_idx) {
                    pass.set_vertex_buffer(0, mesh.vertex_slice());
                    pass.set_index_buffer(mesh.index_slice(), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(
                        0..mesh.index_count(),
                        0,
                        inst_start..inst_start + inst_count,
                    );
                }
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Collect renderable nodes into batches sorted by (material, mesh).
    fn collect_batches(&self, scene: &SceneGraph) -> Vec<DrawBatch> {
        // Collect all renderable items.
        let mut items: Vec<(usize, usize, Mat4)> = scene
            .renderable_nodes()
            .map(|(_handle, mesh_idx, mat_idx, world)| {
                (mat_idx.unwrap_or(usize::MAX), mesh_idx, world)
            })
            .collect();

        if items.is_empty() {
            return Vec::new();
        }

        // Sort by (material, mesh) to minimise bind group changes.
        items.sort_unstable_by_key(|&(mat, mesh, _)| (mat, mesh));

        // Group into batches.
        let mut batches = Vec::new();
        let mut current_mat = items[0].0;
        let mut current_mesh = items[0].1;
        let mut current_instances = Vec::new();

        for (mat, mesh, world) in items {
            if mat != current_mat || mesh != current_mesh {
                batches.push(DrawBatch {
                    material_index: current_mat,
                    mesh_index: current_mesh,
                    instances: std::mem::take(&mut current_instances),
                });
                current_mat = mat;
                current_mesh = mesh;
            }
            current_instances.push(InstanceData::from_mat4(world));
        }

        // Push last batch.
        batches.push(DrawBatch {
            material_index: current_mat,
            mesh_index: current_mesh,
            instances: current_instances,
        });

        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use pac_math::Transform;
    use std::mem;

    // ── SceneUniforms layout ──────────────────────────────────────────

    #[test]
    fn scene_uniforms_size_is_80_bytes() {
        assert_eq!(mem::size_of::<SceneUniforms>(), 80);
    }

    #[test]
    fn scene_uniforms_alignment_is_4() {
        assert_eq!(mem::align_of::<SceneUniforms>(), 4);
    }

    #[test]
    fn scene_uniforms_round_trips_through_bytes() {
        let u = SceneUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [1.0, 2.0, 3.0, 0.0],
        };
        let bytes = bytemuck::bytes_of(&u);
        let back: &SceneUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(back.camera_pos, u.camera_pos);
    }

    // ── SceneUniforms from_camera ──────────────────────────────────────

    #[test]
    fn from_camera_stores_position() {
        let cam = Camera::new(Vec3::new(1.0, 2.0, 3.0), 0.0, 0.0);
        let u = SceneUniforms::from_camera(&cam);
        assert_eq!(u.camera_pos[0], 1.0);
        assert_eq!(u.camera_pos[1], 2.0);
        assert_eq!(u.camera_pos[2], 3.0);
        assert_eq!(u.camera_pos[3], 0.0);
    }

    #[test]
    fn from_camera_stores_view_proj() {
        let cam = Camera::new(Vec3::ZERO, 0.5, 0.3);
        let u = SceneUniforms::from_camera(&cam);
        let expected = cam.view_projection_matrix().to_cols_array_2d();
        assert_eq!(u.view_proj, expected);
    }

    // ── Batch collection ──────────────────────────────────────────────

    #[test]
    fn empty_scene_produces_no_batches() {
        // Renderer::collect_batches only needs &self for method dispatch,
        // but we can't construct Renderer without a GPU device. Test the
        // logic via a standalone helper.
        let scene = SceneGraph::new();
        let batches = collect_batches_standalone(&scene);
        assert!(batches.is_empty());
    }

    #[test]
    fn single_renderable_produces_one_batch() {
        let mut scene = SceneGraph::new();
        let n = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(n).mesh = Some(0);
        scene.node_mut(n).material = Some(0);
        scene.update_world_matrices();

        let batches = collect_batches_standalone(&scene);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].mesh_index, 0);
        assert_eq!(batches[0].material_index, 0);
        assert_eq!(batches[0].instances.len(), 1);
    }

    #[test]
    fn same_material_and_mesh_batched_together() {
        let mut scene = SceneGraph::new();
        for i in 0..5 {
            let n = scene.add_child(
                scene.root(),
                Transform::from_position(Vec3::new(i as f32, 0.0, 0.0)),
            );
            scene.node_mut(n).mesh = Some(0);
            scene.node_mut(n).material = Some(1);
        }
        scene.update_world_matrices();

        let batches = collect_batches_standalone(&scene);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].instances.len(), 5);
    }

    #[test]
    fn different_materials_produce_separate_batches() {
        let mut scene = SceneGraph::new();
        let a = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(a).mesh = Some(0);
        scene.node_mut(a).material = Some(0);

        let b = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(b).mesh = Some(0);
        scene.node_mut(b).material = Some(1);

        scene.update_world_matrices();

        let batches = collect_batches_standalone(&scene);
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn different_meshes_produce_separate_batches() {
        let mut scene = SceneGraph::new();
        let a = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(a).mesh = Some(0);
        scene.node_mut(a).material = Some(0);

        let b = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(b).mesh = Some(1);
        scene.node_mut(b).material = Some(0);

        scene.update_world_matrices();

        let batches = collect_batches_standalone(&scene);
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn batches_sorted_by_material_then_mesh() {
        let mut scene = SceneGraph::new();

        // Add in reverse order.
        let c = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(c).mesh = Some(1);
        scene.node_mut(c).material = Some(2);

        let b = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(b).mesh = Some(0);
        scene.node_mut(b).material = Some(1);

        let a = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(a).mesh = Some(0);
        scene.node_mut(a).material = Some(0);

        scene.update_world_matrices();

        let batches = collect_batches_standalone(&scene);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].material_index, 0);
        assert_eq!(batches[1].material_index, 1);
        assert_eq!(batches[2].material_index, 2);
    }

    #[test]
    fn no_material_nodes_batched_together() {
        let mut scene = SceneGraph::new();
        let a = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(a).mesh = Some(0);
        // material = None

        let b = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(b).mesh = Some(0);
        // material = None

        scene.update_world_matrices();

        let batches = collect_batches_standalone(&scene);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].material_index, usize::MAX);
        assert_eq!(batches[0].instances.len(), 2);
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

        let batches = collect_batches_standalone(&scene);
        let inst = &batches[0].instances[0];
        // Column 3 (translation) should have x=5.0.
        assert!((inst.model[3][0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn group_nodes_without_mesh_are_skipped() {
        let mut scene = SceneGraph::new();
        // Group node (no mesh).
        let group = scene.add_child(scene.root(), Transform::IDENTITY);
        // Renderable child.
        let child = scene.add_child(group, Transform::IDENTITY);
        scene.node_mut(child).mesh = Some(0);
        scene.node_mut(child).material = Some(0);

        scene.update_world_matrices();

        let batches = collect_batches_standalone(&scene);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].instances.len(), 1);
    }

    /// Standalone batch collection matching Renderer::collect_batches logic,
    /// usable without a GPU device.
    fn collect_batches_standalone(scene: &SceneGraph) -> Vec<DrawBatch> {
        let mut items: Vec<(usize, usize, Mat4)> = scene
            .renderable_nodes()
            .map(|(_handle, mesh_idx, mat_idx, world)| {
                (mat_idx.unwrap_or(usize::MAX), mesh_idx, world)
            })
            .collect();

        if items.is_empty() {
            return Vec::new();
        }

        items.sort_unstable_by_key(|&(mat, mesh, _)| (mat, mesh));

        let mut batches = Vec::new();
        let mut current_mat = items[0].0;
        let mut current_mesh = items[0].1;
        let mut current_instances = Vec::new();

        for (mat, mesh, world) in items {
            if mat != current_mat || mesh != current_mesh {
                batches.push(DrawBatch {
                    material_index: current_mat,
                    mesh_index: current_mesh,
                    instances: std::mem::take(&mut current_instances),
                });
                current_mat = mat;
                current_mesh = mesh;
            }
            current_instances.push(InstanceData::from_mat4(world));
        }

        batches.push(DrawBatch {
            material_index: current_mat,
            mesh_index: current_mesh,
            instances: current_instances,
        });

        batches
    }
}
