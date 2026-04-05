//! Pellet system — spawning from maze, collection tracking, instanced rendering.

use glam::Vec3;
use pac_math::Transform;
use pac_render::{wgpu, GpuMesh, InstanceBuffer, InstanceData, Mesh};

use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};

/// Scale factor applied to the unit sphere for normal pellets.
const PELLET_SCALE: f32 = 0.2;
/// Scale factor applied to the unit sphere for power pellets.
const POWER_PELLET_SCALE: f32 = 0.45;
/// Y-axis height at which pellets are placed in world space.
const PELLET_Y: f32 = 0.0;

/// Kind of pellet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PelletKind {
    Normal,
    Power,
}

/// A single pellet on the maze grid.
#[derive(Debug, Clone, Copy)]
pub struct Pellet {
    pub grid_x: usize,
    pub grid_y: usize,
    pub kind: PelletKind,
    pub collected: bool,
}

/// Manages all pellets: spawning from maze, collection tracking, instanced rendering data.
pub struct PelletManager {
    pellets: Vec<Pellet>,
    instances: Vec<InstanceData>,
    dirty: bool,
}

impl PelletManager {
    /// Create a `PelletManager` by scanning the maze for pellet and power-pellet tiles.
    ///
    /// Each `TileType::Pellet` and `TileType::PowerPellet` becomes an entry.
    /// Instance data is built eagerly so it is ready for the first render frame.
    pub fn from_maze(maze: &MazeData) -> Self {
        let mut pellets = Vec::new();
        for y in 0..MAZE_HEIGHT {
            for x in 0..MAZE_WIDTH {
                let kind = match maze.tiles[y][x] {
                    TileType::Pellet => PelletKind::Normal,
                    TileType::PowerPellet => PelletKind::Power,
                    _ => continue,
                };
                pellets.push(Pellet {
                    grid_x: x,
                    grid_y: y,
                    kind,
                    collected: false,
                });
            }
        }
        let mut mgr = Self {
            pellets,
            instances: Vec::new(),
            dirty: true,
        };
        mgr.rebuild_instances();
        mgr
    }

    /// Total number of pellets (including collected).
    pub fn total_count(&self) -> usize {
        self.pellets.len()
    }

    /// Number of pellets not yet collected.
    pub fn remaining_count(&self) -> usize {
        self.pellets.iter().filter(|p| !p.collected).count()
    }

    /// Number of pellets that have been collected.
    pub fn collected_count(&self) -> usize {
        self.pellets.iter().filter(|p| p.collected).count()
    }

    /// Whether all pellets have been collected.
    pub fn all_collected(&self) -> bool {
        self.pellets.iter().all(|p| p.collected)
    }

    /// Collect the pellet at grid position `(x, y)` if present and not yet collected.
    ///
    /// Returns the kind of pellet collected, or `None` if no uncollected pellet exists
    /// at that position.
    pub fn collect_at(&mut self, x: usize, y: usize) -> Option<PelletKind> {
        for pellet in &mut self.pellets {
            if pellet.grid_x == x && pellet.grid_y == y && !pellet.collected {
                pellet.collected = true;
                self.dirty = true;
                return Some(pellet.kind);
            }
        }
        None
    }

    /// Reset all pellets to uncollected state.
    pub fn reset(&mut self) {
        for pellet in &mut self.pellets {
            pellet.collected = false;
        }
        self.dirty = true;
    }

    /// Read-only access to all pellets.
    pub fn pellets(&self) -> &[Pellet] {
        &self.pellets
    }

    /// Instance transforms for uncollected pellets, suitable for instanced draw calls.
    ///
    /// Lazily rebuilds the cache when pellet collection state has changed.
    pub fn instance_data(&mut self) -> &[InstanceData] {
        if self.dirty {
            self.rebuild_instances();
        }
        &self.instances
    }

    /// Create a new GPU instance buffer from the current pellet state.
    pub fn create_instance_buffer(&mut self, device: &wgpu::Device) -> InstanceBuffer {
        let data = self.instance_data();
        InstanceBuffer::new(device, "pellet_instances", data)
    }

    /// Update an existing GPU instance buffer with the current pellet state.
    pub fn update_instance_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut InstanceBuffer,
    ) {
        if self.dirty {
            self.rebuild_instances();
        }
        buffer.write(device, queue, &self.instances);
    }

    /// Create the shared GPU mesh used for all pellet instances (a small sphere).
    pub fn create_mesh(device: &wgpu::Device) -> GpuMesh {
        Mesh::sphere(12, 8).upload(device, "pellet_mesh")
    }

    fn rebuild_instances(&mut self) {
        self.instances.clear();
        for pellet in &self.pellets {
            if pellet.collected {
                continue;
            }
            let scale = match pellet.kind {
                PelletKind::Normal => PELLET_SCALE,
                PelletKind::Power => POWER_PELLET_SCALE,
            };
            let transform = Transform::new(
                Vec3::new(pellet.grid_x as f32, PELLET_Y, pellet.grid_y as f32),
                glam::Quat::IDENTITY,
                Vec3::splat(scale),
            );
            self.instances
                .push(InstanceData::from_mat4(transform.to_matrix()));
        }
        self.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classic_maze() -> MazeData {
        let data = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/maze/classic.json"),
        )
        .expect("classic.json should exist");
        MazeData::from_json(&data).expect("should parse classic maze")
    }

    /// Minimal 28x31 maze with known pellet layout for deterministic tests.
    fn tiny_maze() -> MazeData {
        let mut tiles = [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT];
        // Border walls
        for x in 0..MAZE_WIDTH {
            tiles[0][x] = TileType::Wall;
            tiles[MAZE_HEIGHT - 1][x] = TileType::Wall;
        }
        for y in 0..MAZE_HEIGHT {
            tiles[y][0] = TileType::Wall;
            tiles[y][MAZE_WIDTH - 1] = TileType::Wall;
        }
        // 3 normal pellets + 1 power pellet
        tiles[1][1] = TileType::Pellet;
        tiles[1][2] = TileType::Pellet;
        tiles[1][3] = TileType::PowerPellet;
        tiles[2][1] = TileType::Pellet;
        // Other tile types (should be ignored)
        tiles[15][14] = TileType::PlayerSpawn;
        tiles[13][14] = TileType::GhostHouse;
        MazeData { tiles }
    }

    // ── Spawning from maze ──────────────────────────────────────

    #[test]
    fn from_maze_finds_pellets() {
        let mgr = PelletManager::from_maze(&tiny_maze());
        assert_eq!(mgr.total_count(), 4);
    }

    #[test]
    fn from_maze_correct_kinds() {
        let mgr = PelletManager::from_maze(&tiny_maze());
        let normal = mgr.pellets().iter().filter(|p| p.kind == PelletKind::Normal).count();
        let power = mgr.pellets().iter().filter(|p| p.kind == PelletKind::Power).count();
        assert_eq!(normal, 3);
        assert_eq!(power, 1);
    }

    #[test]
    fn from_maze_ignores_non_pellet_tiles() {
        let mgr = PelletManager::from_maze(&tiny_maze());
        for pellet in mgr.pellets() {
            assert!(
                pellet.kind == PelletKind::Normal || pellet.kind == PelletKind::Power,
                "unexpected pellet kind"
            );
        }
    }

    #[test]
    fn from_maze_stores_grid_positions() {
        let mgr = PelletManager::from_maze(&tiny_maze());
        let positions: Vec<(usize, usize)> =
            mgr.pellets().iter().map(|p| (p.grid_x, p.grid_y)).collect();
        assert!(positions.contains(&(1, 1)));
        assert!(positions.contains(&(2, 1)));
        assert!(positions.contains(&(3, 1)));
        assert!(positions.contains(&(1, 2)));
    }

    #[test]
    fn from_maze_all_uncollected() {
        let mgr = PelletManager::from_maze(&tiny_maze());
        assert!(mgr.pellets().iter().all(|p| !p.collected));
    }

    #[test]
    fn classic_maze_has_pellets() {
        let mgr = PelletManager::from_maze(&classic_maze());
        assert!(mgr.total_count() > 0, "classic maze should have pellets");
    }

    #[test]
    fn classic_maze_has_power_pellets() {
        let mgr = PelletManager::from_maze(&classic_maze());
        let power = mgr.pellets().iter().filter(|p| p.kind == PelletKind::Power).count();
        assert!(power > 0, "classic maze should have power pellets");
    }

    #[test]
    fn empty_maze_produces_no_pellets() {
        let tiles = [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT];
        let mgr = PelletManager::from_maze(&MazeData { tiles });
        assert_eq!(mgr.total_count(), 0);
    }

    // ── Collection tracking ─────────────────────────────────────

    #[test]
    fn collect_at_returns_kind() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        assert_eq!(mgr.collect_at(1, 1), Some(PelletKind::Normal));
        assert_eq!(mgr.collect_at(3, 1), Some(PelletKind::Power));
    }

    #[test]
    fn collect_at_marks_collected() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        mgr.collect_at(1, 1);
        let pellet = mgr.pellets().iter().find(|p| p.grid_x == 1 && p.grid_y == 1).unwrap();
        assert!(pellet.collected);
    }

    #[test]
    fn collect_at_returns_none_for_empty_position() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        assert_eq!(mgr.collect_at(0, 0), None); // wall tile
        assert_eq!(mgr.collect_at(5, 5), None); // empty tile
    }

    #[test]
    fn collect_at_returns_none_when_already_collected() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        mgr.collect_at(1, 1);
        assert_eq!(mgr.collect_at(1, 1), None);
    }

    #[test]
    fn collect_at_out_of_bounds_returns_none() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        assert_eq!(mgr.collect_at(100, 100), None);
    }

    #[test]
    fn remaining_count_decreases_on_collect() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        assert_eq!(mgr.remaining_count(), 4);
        mgr.collect_at(1, 1);
        assert_eq!(mgr.remaining_count(), 3);
        mgr.collect_at(2, 1);
        assert_eq!(mgr.remaining_count(), 2);
    }

    #[test]
    fn collected_count_increases_on_collect() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        assert_eq!(mgr.collected_count(), 0);
        mgr.collect_at(1, 1);
        assert_eq!(mgr.collected_count(), 1);
    }

    #[test]
    fn all_collected_false_initially() {
        let mgr = PelletManager::from_maze(&tiny_maze());
        assert!(!mgr.all_collected());
    }

    #[test]
    fn all_collected_true_when_all_eaten() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        mgr.collect_at(1, 1);
        mgr.collect_at(2, 1);
        mgr.collect_at(3, 1);
        mgr.collect_at(1, 2);
        assert!(mgr.all_collected());
    }

    #[test]
    fn reset_restores_all_pellets() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        mgr.collect_at(1, 1);
        mgr.collect_at(2, 1);
        assert_eq!(mgr.remaining_count(), 2);
        mgr.reset();
        assert_eq!(mgr.remaining_count(), 4);
        assert!(!mgr.all_collected());
    }

    // ── Instance data (rendering) ───────────────────────────────

    #[test]
    fn instance_data_count_matches_remaining() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        assert_eq!(mgr.instance_data().len(), 4);
        mgr.collect_at(1, 1);
        assert_eq!(mgr.instance_data().len(), 3);
    }

    #[test]
    fn instance_data_empty_when_all_collected() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        mgr.collect_at(1, 1);
        mgr.collect_at(2, 1);
        mgr.collect_at(3, 1);
        mgr.collect_at(1, 2);
        assert!(mgr.instance_data().is_empty());
    }

    #[test]
    fn instance_data_rebuilds_after_reset() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        mgr.collect_at(1, 1);
        assert_eq!(mgr.instance_data().len(), 3);
        mgr.reset();
        assert_eq!(mgr.instance_data().len(), 4);
    }

    #[test]
    fn instance_data_has_correct_translations() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        let instances = mgr.instance_data();
        // First pellet is at grid (1, 1) → world (1.0, PELLET_Y, 1.0)
        let mat = glam::Mat4::from_cols_array_2d(&instances[0].model);
        let pos = mat.w_axis.truncate();
        assert!((pos.x - 1.0).abs() < 1e-5);
        assert!((pos.y - PELLET_Y).abs() < 1e-5);
        assert!((pos.z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normal_pellet_scale_smaller_than_power() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        let instances = mgr.instance_data();
        // Index 0: grid (1,1) = Normal pellet
        let normal_mat = glam::Mat4::from_cols_array_2d(&instances[0].model);
        let normal_scale = normal_mat.x_axis.truncate().length();
        // Index 2: grid (3,1) = Power pellet
        let power_mat = glam::Mat4::from_cols_array_2d(&instances[2].model);
        let power_scale = power_mat.x_axis.truncate().length();
        assert!(
            normal_scale < power_scale,
            "normal pellet ({normal_scale}) should be smaller than power pellet ({power_scale})"
        );
    }

    #[test]
    fn pellet_scale_matches_constants() {
        let mut mgr = PelletManager::from_maze(&tiny_maze());
        let instances = mgr.instance_data();
        // Normal pellet at index 0
        let mat = glam::Mat4::from_cols_array_2d(&instances[0].model);
        let scale = mat.x_axis.truncate().length();
        assert!((scale - PELLET_SCALE).abs() < 1e-5);
        // Power pellet at index 2
        let mat = glam::Mat4::from_cols_array_2d(&instances[2].model);
        let scale = mat.x_axis.truncate().length();
        assert!((scale - POWER_PELLET_SCALE).abs() < 1e-5);
    }

    #[test]
    fn instance_data_is_empty_for_empty_maze() {
        let tiles = [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT];
        let mut mgr = PelletManager::from_maze(&MazeData { tiles });
        assert!(mgr.instance_data().is_empty());
    }
}
