//! Scene graph with parent-child node hierarchy and recursive world matrices.
//!
//! Nodes are stored in a flat arena ([`SceneGraph`]) and referenced via
//! lightweight [`NodeHandle`] indices. Each [`SceneNode`] carries a local
//! [`Transform`](pac_math::Transform), optional mesh and material indices,
//! and a list of child handles.
//!
//! Call [`SceneGraph::update_world_matrices`] once per frame to propagate
//! transforms down the hierarchy. The resulting world matrix for each node
//! is cached and accessible via [`SceneGraph::world_matrix`].

use glam::Mat4;
use pac_math::Transform;

// ── NodeHandle ──────────────────────────────────────────────────────────

/// Lightweight handle into the scene graph's node arena.
///
/// Internally a `usize` index. Handles are only valid for the graph that
/// created them — using a handle from one graph in another is a logic error
/// (will panic or return wrong data).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeHandle(usize);

impl NodeHandle {
    /// Raw index into the arena (useful for external parallel arrays).
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

// ── SceneNode ───────────────────────────────────────────────────────────

/// A single node in the scene graph.
///
/// Each node has a local transform relative to its parent, optional indices
/// into external mesh/material arrays, and a list of child handles.
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub transform: Transform,
    /// Index into an external mesh array, or `None` for group-only nodes.
    pub mesh: Option<usize>,
    /// Index into an external material array, or `None` for default material.
    pub material: Option<usize>,
    pub children: Vec<NodeHandle>,
    parent: Option<NodeHandle>,
}

impl SceneNode {
    fn new(transform: Transform) -> Self {
        Self {
            transform,
            mesh: None,
            material: None,
            children: Vec::new(),
            parent: None,
        }
    }
}

// ── SceneGraph ──────────────────────────────────────────────────────────

/// Arena-based scene graph with cached world matrices.
///
/// Nodes are stored in a flat `Vec` and referenced by [`NodeHandle`].
/// A dedicated root node (handle 0) is created automatically; attach
/// top-level objects as children of the root.
pub struct SceneGraph {
    nodes: Vec<SceneNode>,
    world_matrices: Vec<Mat4>,
}

impl SceneGraph {
    /// Create a new scene graph with an identity root node.
    pub fn new() -> Self {
        let root = SceneNode::new(Transform::IDENTITY);
        Self {
            nodes: vec![root],
            world_matrices: vec![Mat4::IDENTITY],
        }
    }

    /// Handle to the root node (always index 0).
    #[inline]
    pub fn root(&self) -> NodeHandle {
        NodeHandle(0)
    }

    /// Total number of nodes (including root).
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph contains only the root node.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 1
    }

    /// Add a new node with the given transform, returning its handle.
    ///
    /// The node starts with no parent, no mesh, and no material.
    /// Use [`Self::set_parent`] or [`Self::add_child`] to place it in the
    /// hierarchy.
    pub fn add_node(&mut self, transform: Transform) -> NodeHandle {
        let handle = NodeHandle(self.nodes.len());
        self.nodes.push(SceneNode::new(transform));
        self.world_matrices.push(Mat4::IDENTITY);
        handle
    }

    /// Attach `child` as a child of `parent`.
    ///
    /// If `child` already has a different parent, it is removed from the old
    /// parent's children list first.
    pub fn set_parent(&mut self, child: NodeHandle, parent: NodeHandle) {
        assert_ne!(child, parent, "a node cannot be its own parent");

        // Remove from old parent if any.
        if let Some(old_parent) = self.nodes[child.0].parent {
            self.nodes[old_parent.0]
                .children
                .retain(|h| *h != child);
        }

        self.nodes[child.0].parent = Some(parent);
        self.nodes[parent.0].children.push(child);
    }

    /// Convenience: create a node and immediately attach it as a child.
    pub fn add_child(
        &mut self,
        parent: NodeHandle,
        transform: Transform,
    ) -> NodeHandle {
        let handle = self.add_node(transform);
        self.set_parent(handle, parent);
        handle
    }

    /// Read-only access to a node.
    #[inline]
    pub fn node(&self, handle: NodeHandle) -> &SceneNode {
        &self.nodes[handle.0]
    }

    /// Mutable access to a node (e.g. to change its transform or mesh).
    #[inline]
    pub fn node_mut(&mut self, handle: NodeHandle) -> &mut SceneNode {
        &mut self.nodes[handle.0]
    }

    /// The cached world matrix for `handle`.
    ///
    /// Only valid after the most recent [`Self::update_world_matrices`] call.
    #[inline]
    pub fn world_matrix(&self, handle: NodeHandle) -> Mat4 {
        self.world_matrices[handle.0]
    }

    /// Recursively recompute all world matrices starting from the root.
    ///
    /// Call once per frame after modifying any transforms. The root's local
    /// transform is applied as-is (its parent matrix is identity).
    pub fn update_world_matrices(&mut self) {
        self.update_recursive(NodeHandle(0), Mat4::IDENTITY);
    }

    fn update_recursive(&mut self, handle: NodeHandle, parent_world: Mat4) {
        let local = self.nodes[handle.0].transform.to_matrix();
        let world = parent_world * local;
        self.world_matrices[handle.0] = world;

        // Collect children to avoid borrow conflict.
        let children: Vec<NodeHandle> = self.nodes[handle.0].children.clone();
        for child in children {
            self.update_recursive(child, world);
        }
    }

    /// Iterate over all nodes that have a mesh attached, yielding
    /// `(handle, mesh_index, material_index_option, world_matrix)`.
    ///
    /// Useful for building draw call lists. Only valid after
    /// [`Self::update_world_matrices`].
    pub fn renderable_nodes(&self) -> impl Iterator<Item = (NodeHandle, usize, Option<usize>, Mat4)> + '_ {
        self.nodes.iter().enumerate().filter_map(|(i, node)| {
            node.mesh.map(|mesh_idx| {
                (NodeHandle(i), mesh_idx, node.material, self.world_matrices[i])
            })
        })
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};
    use std::f32::consts::FRAC_PI_2;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    // ── Construction ──────────────────────────────────────────────

    #[test]
    fn new_graph_has_root() {
        let g = SceneGraph::new();
        assert_eq!(g.len(), 1);
        assert_eq!(g.root(), NodeHandle(0));
    }

    #[test]
    fn new_graph_is_empty() {
        let g = SceneGraph::new();
        assert!(g.is_empty());
    }

    #[test]
    fn default_creates_same_as_new() {
        let g = SceneGraph::default();
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn add_node_returns_sequential_handles() {
        let mut g = SceneGraph::new();
        let a = g.add_node(Transform::IDENTITY);
        let b = g.add_node(Transform::IDENTITY);
        assert_eq!(a.index(), 1);
        assert_eq!(b.index(), 2);
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn add_node_not_empty() {
        let mut g = SceneGraph::new();
        g.add_node(Transform::IDENTITY);
        assert!(!g.is_empty());
    }

    // ── Parent/child ──────────────────────────────────────────────

    #[test]
    fn set_parent_links_child() {
        let mut g = SceneGraph::new();
        let child = g.add_node(Transform::IDENTITY);
        g.set_parent(child, g.root());
        assert_eq!(g.node(g.root()).children.len(), 1);
        assert_eq!(g.node(g.root()).children[0], child);
    }

    #[test]
    fn add_child_convenience() {
        let mut g = SceneGraph::new();
        let child = g.add_child(g.root(), Transform::IDENTITY);
        assert_eq!(g.node(g.root()).children.len(), 1);
        assert_eq!(g.node(g.root()).children[0], child);
    }

    #[test]
    fn reparent_removes_from_old_parent() {
        let mut g = SceneGraph::new();
        let a = g.add_child(g.root(), Transform::IDENTITY);
        let b = g.add_child(g.root(), Transform::IDENTITY);
        assert_eq!(g.node(g.root()).children.len(), 2);

        // Move b under a
        g.set_parent(b, a);
        assert_eq!(g.node(g.root()).children.len(), 1);
        assert_eq!(g.node(a).children.len(), 1);
        assert_eq!(g.node(a).children[0], b);
    }

    #[test]
    #[should_panic(expected = "a node cannot be its own parent")]
    fn set_parent_self_panics() {
        let mut g = SceneGraph::new();
        let n = g.add_node(Transform::IDENTITY);
        g.set_parent(n, n);
    }

    // ── Node access ──────────────────────────────────────────────

    #[test]
    fn node_mut_allows_modification() {
        let mut g = SceneGraph::new();
        let n = g.add_child(g.root(), Transform::IDENTITY);
        g.node_mut(n).mesh = Some(42);
        g.node_mut(n).material = Some(7);
        assert_eq!(g.node(n).mesh, Some(42));
        assert_eq!(g.node(n).material, Some(7));
    }

    #[test]
    fn node_default_has_no_mesh_or_material() {
        let mut g = SceneGraph::new();
        let n = g.add_node(Transform::IDENTITY);
        assert_eq!(g.node(n).mesh, None);
        assert_eq!(g.node(n).material, None);
    }

    // ── World matrix propagation ────────────────────────────────

    #[test]
    fn identity_hierarchy_produces_identity_matrices() {
        let mut g = SceneGraph::new();
        let a = g.add_child(g.root(), Transform::IDENTITY);
        let _b = g.add_child(a, Transform::IDENTITY);
        g.update_world_matrices();
        assert_eq!(g.world_matrix(g.root()), Mat4::IDENTITY);
        assert_eq!(g.world_matrix(a), Mat4::IDENTITY);
        assert_eq!(g.world_matrix(_b), Mat4::IDENTITY);
    }

    #[test]
    fn translation_propagates_to_children() {
        let mut g = SceneGraph::new();
        let parent = g.add_child(
            g.root(),
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
        );
        let child = g.add_child(
            parent,
            Transform::from_position(Vec3::new(0.0, 5.0, 0.0)),
        );
        g.update_world_matrices();

        let child_world = g.world_matrix(child);
        // Child should be at (10, 5, 0) in world space.
        let pos = child_world.transform_point3(Vec3::ZERO);
        assert!(approx_eq(pos.x, 10.0));
        assert!(approx_eq(pos.y, 5.0));
        assert!(approx_eq(pos.z, 0.0));
    }

    #[test]
    fn scale_propagates_to_children() {
        let mut g = SceneGraph::new();
        let parent = g.add_child(
            g.root(),
            Transform::from_scale(Vec3::new(2.0, 2.0, 2.0)),
        );
        let child = g.add_child(
            parent,
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
        );
        g.update_world_matrices();

        let pos = g.world_matrix(child).transform_point3(Vec3::ZERO);
        // Parent scales by 2, child is at local (1,0,0) -> world (2,0,0).
        assert!(approx_eq(pos.x, 2.0));
        assert!(approx_eq(pos.y, 0.0));
        assert!(approx_eq(pos.z, 0.0));
    }

    #[test]
    fn rotation_propagates_to_children() {
        let mut g = SceneGraph::new();
        // Parent rotates 90° around Y.
        let parent = g.add_child(
            g.root(),
            Transform::from_rotation(Quat::from_rotation_y(FRAC_PI_2)),
        );
        // Child offset along local +X.
        let child = g.add_child(
            parent,
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
        );
        g.update_world_matrices();

        let pos = g.world_matrix(child).transform_point3(Vec3::ZERO);
        // 90° Y rotation maps local +X to world -Z.
        assert!(approx_eq(pos.x, 0.0));
        assert!(approx_eq(pos.y, 0.0));
        assert!(approx_eq(pos.z, -1.0));
    }

    #[test]
    fn three_level_hierarchy() {
        let mut g = SceneGraph::new();
        let a = g.add_child(
            g.root(),
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
        );
        let b = g.add_child(
            a,
            Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
        );
        let c = g.add_child(
            b,
            Transform::from_position(Vec3::new(0.0, 0.0, 1.0)),
        );
        g.update_world_matrices();

        let pos = g.world_matrix(c).transform_point3(Vec3::ZERO);
        assert!(approx_eq(pos.x, 1.0));
        assert!(approx_eq(pos.y, 1.0));
        assert!(approx_eq(pos.z, 1.0));
    }

    #[test]
    fn multiple_children_get_independent_world_matrices() {
        let mut g = SceneGraph::new();
        let parent = g.add_child(
            g.root(),
            Transform::from_position(Vec3::new(5.0, 0.0, 0.0)),
        );
        let c1 = g.add_child(
            parent,
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
        );
        let c2 = g.add_child(
            parent,
            Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
        );
        g.update_world_matrices();

        let p1 = g.world_matrix(c1).transform_point3(Vec3::ZERO);
        let p2 = g.world_matrix(c2).transform_point3(Vec3::ZERO);
        assert!(approx_eq(p1.x, 6.0));
        assert!(approx_eq(p1.y, 0.0));
        assert!(approx_eq(p2.x, 5.0));
        assert!(approx_eq(p2.y, 1.0));
    }

    #[test]
    fn update_after_transform_change() {
        let mut g = SceneGraph::new();
        let n = g.add_child(
            g.root(),
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
        );
        g.update_world_matrices();
        let p1 = g.world_matrix(n).transform_point3(Vec3::ZERO);
        assert!(approx_eq(p1.x, 1.0));

        // Change transform.
        g.node_mut(n).transform.position = Vec3::new(3.0, 0.0, 0.0);
        g.update_world_matrices();
        let p2 = g.world_matrix(n).transform_point3(Vec3::ZERO);
        assert!(approx_eq(p2.x, 3.0));
    }

    #[test]
    fn root_transform_applies() {
        let mut g = SceneGraph::new();
        g.node_mut(g.root()).transform = Transform::from_position(Vec3::new(0.0, 100.0, 0.0));
        let child = g.add_child(g.root(), Transform::IDENTITY);
        g.update_world_matrices();

        let pos = g.world_matrix(child).transform_point3(Vec3::ZERO);
        assert!(approx_eq(pos.y, 100.0));
    }

    // ── Renderable nodes ────────────────────────────────────────

    #[test]
    fn renderable_nodes_empty_when_no_meshes() {
        let mut g = SceneGraph::new();
        g.add_child(g.root(), Transform::IDENTITY);
        g.update_world_matrices();
        assert_eq!(g.renderable_nodes().count(), 0);
    }

    #[test]
    fn renderable_nodes_returns_mesh_nodes() {
        let mut g = SceneGraph::new();
        let a = g.add_child(g.root(), Transform::IDENTITY);
        g.node_mut(a).mesh = Some(0);
        g.node_mut(a).material = Some(1);
        let b = g.add_child(g.root(), Transform::IDENTITY);
        // b has no mesh — should not appear.
        let _ = b;
        g.update_world_matrices();

        let renderables: Vec<_> = g.renderable_nodes().collect();
        assert_eq!(renderables.len(), 1);
        assert_eq!(renderables[0].0, a);
        assert_eq!(renderables[0].1, 0); // mesh index
        assert_eq!(renderables[0].2, Some(1)); // material index
    }

    #[test]
    fn renderable_nodes_respects_world_matrix() {
        let mut g = SceneGraph::new();
        let n = g.add_child(
            g.root(),
            Transform::from_position(Vec3::new(5.0, 0.0, 0.0)),
        );
        g.node_mut(n).mesh = Some(0);
        g.update_world_matrices();

        let (_, _, _, world) = g.renderable_nodes().next().unwrap();
        let pos = world.transform_point3(Vec3::ZERO);
        assert!(approx_eq(pos.x, 5.0));
    }

    // ── NodeHandle ──────────────────────────────────────────────

    #[test]
    fn node_handle_index() {
        let h = NodeHandle(42);
        assert_eq!(h.index(), 42);
    }

    #[test]
    fn node_handle_equality() {
        assert_eq!(NodeHandle(1), NodeHandle(1));
        assert_ne!(NodeHandle(1), NodeHandle(2));
    }

    #[test]
    fn node_handle_copy() {
        let a = NodeHandle(5);
        let b = a; // Copy
        assert_eq!(a, b);
    }
}
