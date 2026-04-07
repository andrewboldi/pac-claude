//! Maze 3D renderer — converts [`MazeData`] to scene graph nodes.
//!
//! Walls become scaled cubes, ghost house tiles get a distinct material,
//! ghost doors are shorter cubes, and the floor is a plane spanning the
//! full grid dimensions.

use glam::Vec3;
use pac_math::Transform;
use pac_render::scene::{NodeHandle, SceneGraph};

use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};

/// Wall cube height in world units.
const WALL_HEIGHT: f32 = 1.0;
/// Y position of wall cube centres.
const WALL_Y: f32 = 0.0;
/// Y position of the floor plane (half a wall height below centre).
const FLOOR_Y: f32 = -0.5;
/// Ghost-door cube height (shorter than walls so players can see inside).
const GHOST_DOOR_HEIGHT: f32 = 0.3;
/// Ghost-door Y position (sits at floor level).
const GHOST_DOOR_Y: f32 = FLOOR_Y + GHOST_DOOR_HEIGHT / 2.0;

/// Indices into external mesh / material arrays used when building the maze.
pub struct MazeMeshConfig {
    /// Index of a unit cube mesh (walls, ghost house, ghost door).
    pub wall_mesh: usize,
    /// Index of a unit plane mesh (floor).
    pub floor_mesh: usize,
    /// Material index for maze walls.
    pub wall_material: usize,
    /// Material index for the floor.
    pub floor_material: usize,
    /// Material index for ghost house interior walls.
    pub ghost_house_material: usize,
    /// Material index for the ghost house door.
    pub ghost_door_material: usize,
}

/// Handles produced by [`build_maze_scene`].
pub struct MazeScene {
    /// Root group node for the entire maze (child of the scene-graph root).
    pub root: NodeHandle,
    /// One handle per wall-tile cube.
    pub walls: Vec<NodeHandle>,
    /// One handle per ghost-house-tile cube.
    pub ghost_house: Vec<NodeHandle>,
    /// Handle to the ghost-door cube (if present in the maze).
    pub ghost_door: Option<NodeHandle>,
    /// Handle to the floor plane.
    pub floor: NodeHandle,
}

/// Build 3D scene nodes from maze data and attach them to `scene`.
///
/// Creates a hierarchy under the scene-graph root:
///
/// ```text
/// scene root
/// └─ maze root (group)
///    ├─ wall cubes ...
///    ├─ ghost house cubes ...
///    ├─ ghost door (short cube)
///    └─ floor plane
/// ```
///
/// Grid position `(x, y)` maps to world `(x, WALL_Y, y)`, matching the
/// convention used by [`PelletManager`](crate::pellet::PelletManager).
pub fn build_maze_scene(
    scene: &mut SceneGraph,
    maze: &MazeData,
    config: &MazeMeshConfig,
) -> MazeScene {
    let maze_root = scene.add_child(scene.root(), Transform::IDENTITY);

    let mut walls = Vec::new();
    let mut ghost_house = Vec::new();
    let mut ghost_door = None;

    for y in 0..MAZE_HEIGHT {
        for x in 0..MAZE_WIDTH {
            match maze.tiles[y][x] {
                TileType::Wall => {
                    let h = add_cube(
                        scene,
                        maze_root,
                        x as f32,
                        WALL_Y,
                        y as f32,
                        1.0,
                        WALL_HEIGHT,
                        1.0,
                        config.wall_mesh,
                        config.wall_material,
                    );
                    walls.push(h);
                }
                TileType::GhostHouse => {
                    let h = add_cube(
                        scene,
                        maze_root,
                        x as f32,
                        WALL_Y,
                        y as f32,
                        1.0,
                        WALL_HEIGHT,
                        1.0,
                        config.wall_mesh,
                        config.ghost_house_material,
                    );
                    ghost_house.push(h);
                }
                TileType::GhostDoor => {
                    let h = add_cube(
                        scene,
                        maze_root,
                        x as f32,
                        GHOST_DOOR_Y,
                        y as f32,
                        1.0,
                        GHOST_DOOR_HEIGHT,
                        1.0,
                        config.wall_mesh,
                        config.ghost_door_material,
                    );
                    ghost_door = Some(h);
                }
                _ => {}
            }
        }
    }

    // Floor plane centred under the maze, scaled to cover the full grid.
    let floor_x = (MAZE_WIDTH as f32 - 1.0) / 2.0;
    let floor_z = (MAZE_HEIGHT as f32 - 1.0) / 2.0;
    let floor = scene.add_child(
        maze_root,
        Transform::new(
            Vec3::new(floor_x, FLOOR_Y, floor_z),
            glam::Quat::IDENTITY,
            Vec3::new(MAZE_WIDTH as f32, 1.0, MAZE_HEIGHT as f32),
        ),
    );
    scene.node_mut(floor).mesh = Some(config.floor_mesh);
    scene.node_mut(floor).material = Some(config.floor_material);

    MazeScene {
        root: maze_root,
        walls,
        ghost_house,
        ghost_door,
        floor,
    }
}

fn add_cube(
    scene: &mut SceneGraph,
    parent: NodeHandle,
    x: f32,
    y: f32,
    z: f32,
    sx: f32,
    sy: f32,
    sz: f32,
    mesh: usize,
    material: usize,
) -> NodeHandle {
    let h = scene.add_child(
        parent,
        Transform::new(
            Vec3::new(x, y, z),
            glam::Quat::IDENTITY,
            Vec3::new(sx, sy, sz),
        ),
    );
    scene.node_mut(h).mesh = Some(mesh);
    scene.node_mut(h).material = Some(material);
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};

    fn default_config() -> MazeMeshConfig {
        MazeMeshConfig {
            wall_mesh: 0,
            floor_mesh: 1,
            wall_material: 0,
            floor_material: 1,
            ghost_house_material: 2,
            ghost_door_material: 3,
        }
    }

    fn classic_maze() -> MazeData {
        let data = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/maze/classic.json"),
        )
        .expect("classic.json should exist");
        MazeData::from_json(&data).expect("should parse classic maze")
    }

    fn tiny_maze() -> MazeData {
        let mut tiles = [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT];
        for x in 0..MAZE_WIDTH {
            tiles[0][x] = TileType::Wall;
            tiles[MAZE_HEIGHT - 1][x] = TileType::Wall;
        }
        for y in 0..MAZE_HEIGHT {
            tiles[y][0] = TileType::Wall;
            tiles[y][MAZE_WIDTH - 1] = TileType::Wall;
        }
        tiles[1][1] = TileType::Pellet;
        tiles[1][2] = TileType::Pellet;
        tiles[15][14] = TileType::PlayerSpawn;
        tiles[13][13] = TileType::GhostHouse;
        tiles[13][14] = TileType::GhostHouse;
        tiles[12][13] = TileType::GhostDoor;
        MazeData { tiles }
    }

    fn empty_maze() -> MazeData {
        MazeData {
            tiles: [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT],
        }
    }

    fn wall_count(maze: &MazeData) -> usize {
        maze.tiles
            .iter()
            .flat_map(|r| r.iter())
            .filter(|t| **t == TileType::Wall)
            .count()
    }

    fn ghost_house_count(maze: &MazeData) -> usize {
        maze.tiles
            .iter()
            .flat_map(|r| r.iter())
            .filter(|t| **t == TileType::GhostHouse)
            .count()
    }

    // ── Scene structure ──────────────────────────────────────────

    #[test]
    fn builds_scene_under_root() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        // Maze root is a child of the scene root.
        assert_eq!(scene.node(scene.root()).children.len(), 1);
        assert_eq!(scene.node(scene.root()).children[0], ms.root);
    }

    #[test]
    fn wall_count_matches_tiles() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        assert_eq!(ms.walls.len(), wall_count(&maze));
    }

    #[test]
    fn ghost_house_count_matches_tiles() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        assert_eq!(ms.ghost_house.len(), ghost_house_count(&maze));
    }

    #[test]
    fn ghost_door_present_when_in_maze() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        assert!(ms.ghost_door.is_some());
    }

    #[test]
    fn ghost_door_absent_when_not_in_maze() {
        let mut scene = SceneGraph::new();
        let maze = empty_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        assert!(ms.ghost_door.is_none());
    }

    #[test]
    fn floor_node_exists() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        assert_eq!(scene.node(ms.floor).mesh, Some(1));
        assert_eq!(scene.node(ms.floor).material, Some(1));
    }

    // ── Empty maze ───────────────────────────────────────────────

    #[test]
    fn empty_maze_produces_no_walls() {
        let mut scene = SceneGraph::new();
        let ms = build_maze_scene(&mut scene, &empty_maze(), &default_config());
        assert!(ms.walls.is_empty());
        assert!(ms.ghost_house.is_empty());
        assert!(ms.ghost_door.is_none());
    }

    #[test]
    fn empty_maze_still_has_floor() {
        let mut scene = SceneGraph::new();
        let ms = build_maze_scene(&mut scene, &empty_maze(), &default_config());
        assert_eq!(scene.node(ms.floor).mesh, Some(1));
    }

    // ── Mesh / material assignment ───────────────────────────────

    #[test]
    fn wall_nodes_use_wall_mesh_and_material() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let cfg = default_config();
        let ms = build_maze_scene(&mut scene, &maze, &cfg);
        for &h in &ms.walls {
            assert_eq!(scene.node(h).mesh, Some(cfg.wall_mesh));
            assert_eq!(scene.node(h).material, Some(cfg.wall_material));
        }
    }

    #[test]
    fn ghost_house_nodes_use_ghost_material() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let cfg = default_config();
        let ms = build_maze_scene(&mut scene, &maze, &cfg);
        for &h in &ms.ghost_house {
            assert_eq!(scene.node(h).mesh, Some(cfg.wall_mesh));
            assert_eq!(scene.node(h).material, Some(cfg.ghost_house_material));
        }
    }

    #[test]
    fn ghost_door_uses_door_material() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let cfg = default_config();
        let ms = build_maze_scene(&mut scene, &maze, &cfg);
        let door = ms.ghost_door.unwrap();
        assert_eq!(scene.node(door).mesh, Some(cfg.wall_mesh));
        assert_eq!(scene.node(door).material, Some(cfg.ghost_door_material));
    }

    // ── Transform positions ──────────────────────────────────────

    #[test]
    fn wall_positions_match_grid() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        scene.update_world_matrices();
        // First wall in the maze is at grid (0, 0) → world (0, WALL_Y, 0).
        let world = scene.world_matrix(ms.walls[0]);
        let pos = world.transform_point3(Vec3::ZERO);
        assert!((pos.x - 0.0).abs() < 1e-5);
        assert!((pos.y - WALL_Y).abs() < 1e-5);
        assert!((pos.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn floor_position_centred() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        let t = &scene.node(ms.floor).transform;
        let expected_x = (MAZE_WIDTH as f32 - 1.0) / 2.0;
        let expected_z = (MAZE_HEIGHT as f32 - 1.0) / 2.0;
        assert!((t.position.x - expected_x).abs() < 1e-5);
        assert!((t.position.y - FLOOR_Y).abs() < 1e-5);
        assert!((t.position.z - expected_z).abs() < 1e-5);
    }

    #[test]
    fn floor_scale_covers_maze() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        let s = scene.node(ms.floor).transform.scale;
        assert!((s.x - MAZE_WIDTH as f32).abs() < 1e-5);
        assert!((s.z - MAZE_HEIGHT as f32).abs() < 1e-5);
    }

    #[test]
    fn ghost_door_is_shorter_than_wall() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        let door = ms.ghost_door.unwrap();
        let door_scale_y = scene.node(door).transform.scale.y;
        let wall_scale_y = scene.node(ms.walls[0]).transform.scale.y;
        assert!(
            door_scale_y < wall_scale_y,
            "ghost door ({door_scale_y}) should be shorter than wall ({wall_scale_y})"
        );
    }

    // ── Renderable node count ────────────────────────────────────

    #[test]
    fn renderable_count_equals_walls_plus_ghost_plus_door_plus_floor() {
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        scene.update_world_matrices();
        let expected = ms.walls.len()
            + ms.ghost_house.len()
            + ms.ghost_door.iter().count()
            + 1; // floor
        assert_eq!(scene.renderable_nodes().count(), expected);
    }

    // ── Classic maze ─────────────────────────────────────────────

    #[test]
    fn classic_maze_wall_count() {
        let mut scene = SceneGraph::new();
        let maze = classic_maze();
        let expected = wall_count(&maze);
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        assert_eq!(ms.walls.len(), expected);
    }

    #[test]
    fn classic_maze_has_ghost_house() {
        let mut scene = SceneGraph::new();
        let maze = classic_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        assert!(
            !ms.ghost_house.is_empty(),
            "classic maze should produce ghost house nodes"
        );
    }

    #[test]
    fn classic_maze_has_ghost_door() {
        let mut scene = SceneGraph::new();
        let maze = classic_maze();
        let ms = build_maze_scene(&mut scene, &maze, &default_config());
        assert!(
            ms.ghost_door.is_some(),
            "classic maze should have a ghost door"
        );
    }

    // ── Custom config ────────────────────────────────────────────

    #[test]
    fn custom_config_indices_propagate() {
        let cfg = MazeMeshConfig {
            wall_mesh: 5,
            floor_mesh: 6,
            wall_material: 10,
            floor_material: 11,
            ghost_house_material: 12,
            ghost_door_material: 13,
        };
        let mut scene = SceneGraph::new();
        let maze = tiny_maze();
        let ms = build_maze_scene(&mut scene, &maze, &cfg);
        assert_eq!(scene.node(ms.walls[0]).mesh, Some(5));
        assert_eq!(scene.node(ms.walls[0]).material, Some(10));
        assert_eq!(scene.node(ms.floor).mesh, Some(6));
        assert_eq!(scene.node(ms.floor).material, Some(11));
        assert_eq!(scene.node(ms.ghost_house[0]).material, Some(12));
        assert_eq!(
            scene.node(ms.ghost_door.unwrap()).material,
            Some(13)
        );
    }
}
