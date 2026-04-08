//! Game collision system — Pac-Man vs pellets and ghosts.
//!
//! Wall collision is handled at the movement level in [`crate::pacman`].
//! This module covers entity-vs-entity and entity-vs-pickup collisions
//! that occur after movement has been resolved.

use crate::pacman::PacMan;
use crate::pellet::{PelletKind, PelletManager};

/// Collision state for a single ghost, provided by the ghost system.
///
/// The collision system only needs to know a ghost's grid position and
/// whether it is frightened (edible). The ghost AI module (when implemented)
/// is responsible for populating these values each tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GhostCollider {
    /// Current grid column.
    pub grid_x: usize,
    /// Current grid row.
    pub grid_y: usize,
    /// Whether this ghost is in frightened (edible) mode.
    pub frightened: bool,
}

/// Events produced by collision checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionEvent {
    /// Pac-Man collected a normal pellet at `(x, y)`.
    PelletCollected { x: usize, y: usize },
    /// Pac-Man collected a power pellet at `(x, y)`.
    PowerPelletCollected { x: usize, y: usize },
    /// Pac-Man was hit by ghost `ghost_index` (player loses a life).
    GhostHit { ghost_index: usize },
    /// Pac-Man ate frightened ghost `ghost_index`.
    GhostEaten { ghost_index: usize },
}

/// Check all collisions for the current frame.
///
/// Compares Pac-Man's current grid position against pellets and ghosts,
/// returning a list of collision events. Call this once per fixed-update
/// tick, after [`PacMan::update`] has been called.
///
/// The caller is responsible for applying game-logic consequences
/// (scoring, life loss, ghost reset, etc.).
pub fn check_collisions(
    pacman: &PacMan,
    pellets: &mut PelletManager,
    ghosts: &[GhostCollider],
) -> Vec<CollisionEvent> {
    let mut events = Vec::new();

    let px = pacman.grid_x();
    let py = pacman.grid_y();

    // Pellet collision — collect if Pac-Man's tile has an uncollected pellet.
    if let Some(kind) = pellets.collect_at(px, py) {
        match kind {
            PelletKind::Normal => events.push(CollisionEvent::PelletCollected { x: px, y: py }),
            PelletKind::Power => {
                events.push(CollisionEvent::PowerPelletCollected { x: px, y: py });
            }
        }
    }

    // Ghost collision — same-tile overlap check.
    for (i, ghost) in ghosts.iter().enumerate() {
        if ghost.grid_x == px && ghost.grid_y == py {
            if ghost.frightened {
                events.push(CollisionEvent::GhostEaten { ghost_index: i });
            } else {
                events.push(CollisionEvent::GhostHit { ghost_index: i });
            }
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};
    use crate::pacman::PacMan;
    use crate::pellet::PelletManager;

    /// Maze with a few pellets and open corridors for testing.
    fn collision_maze() -> MazeData {
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
        // Pellets
        tiles[1][1] = TileType::Pellet;
        tiles[1][2] = TileType::Pellet;
        tiles[1][3] = TileType::PowerPellet;
        tiles[2][1] = TileType::Pellet;
        // Player spawn (not a pellet)
        tiles[3][1] = TileType::PlayerSpawn;
        MazeData { tiles }
    }

    // ── Pellet collision ───────────────────────────────────────

    #[test]
    fn collects_normal_pellet() {
        let maze = collision_maze();
        let pac = PacMan::new(1, 1);
        let mut pellets = PelletManager::from_maze(&maze);

        let events = check_collisions(&pac, &mut pellets, &[]);

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            CollisionEvent::PelletCollected { x: 1, y: 1 }
        );
    }

    #[test]
    fn collects_power_pellet() {
        let maze = collision_maze();
        let pac = PacMan::new(3, 1);
        let mut pellets = PelletManager::from_maze(&maze);

        let events = check_collisions(&pac, &mut pellets, &[]);

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            CollisionEvent::PowerPelletCollected { x: 3, y: 1 }
        );
    }

    #[test]
    fn no_event_on_empty_tile() {
        let maze = collision_maze();
        let pac = PacMan::new(5, 5); // empty tile
        let mut pellets = PelletManager::from_maze(&maze);

        let events = check_collisions(&pac, &mut pellets, &[]);

        assert!(events.is_empty());
    }

    #[test]
    fn no_event_on_already_collected_pellet() {
        let maze = collision_maze();
        let pac = PacMan::new(1, 1);
        let mut pellets = PelletManager::from_maze(&maze);

        // First call collects it.
        let events = check_collisions(&pac, &mut pellets, &[]);
        assert_eq!(events.len(), 1);

        // Second call on same tile — already collected.
        let events = check_collisions(&pac, &mut pellets, &[]);
        assert!(events.is_empty());
    }

    #[test]
    fn pellet_remaining_count_decreases() {
        let maze = collision_maze();
        let mut pellets = PelletManager::from_maze(&maze);
        assert_eq!(pellets.remaining_count(), 4);

        let pac = PacMan::new(1, 1);
        check_collisions(&pac, &mut pellets, &[]);
        assert_eq!(pellets.remaining_count(), 3);

        let pac = PacMan::new(1, 2);
        check_collisions(&pac, &mut pellets, &[]);
        assert_eq!(pellets.remaining_count(), 2);
    }

    #[test]
    fn collect_all_pellets_sequentially() {
        let maze = collision_maze();
        let mut pellets = PelletManager::from_maze(&maze);

        let positions = [(1, 1), (2, 1), (3, 1), (1, 2)];
        for &(x, y) in &positions {
            let pac = PacMan::new(x, y);
            let events = check_collisions(&pac, &mut pellets, &[]);
            assert_eq!(events.len(), 1);
        }
        assert!(pellets.all_collected());
    }

    // ── Ghost collision ────────────────────────────────────────

    #[test]
    fn ghost_hit_on_same_tile() {
        let maze = collision_maze();
        let pac = PacMan::new(5, 5);
        let mut pellets = PelletManager::from_maze(&maze);
        let ghosts = [GhostCollider {
            grid_x: 5,
            grid_y: 5,
            frightened: false,
        }];

        let events = check_collisions(&pac, &mut pellets, &ghosts);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0], CollisionEvent::GhostHit { ghost_index: 0 });
    }

    #[test]
    fn ghost_eaten_when_frightened() {
        let maze = collision_maze();
        let pac = PacMan::new(5, 5);
        let mut pellets = PelletManager::from_maze(&maze);
        let ghosts = [GhostCollider {
            grid_x: 5,
            grid_y: 5,
            frightened: true,
        }];

        let events = check_collisions(&pac, &mut pellets, &ghosts);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0], CollisionEvent::GhostEaten { ghost_index: 0 });
    }

    #[test]
    fn no_ghost_event_on_different_tile() {
        let maze = collision_maze();
        let pac = PacMan::new(5, 5);
        let mut pellets = PelletManager::from_maze(&maze);
        let ghosts = [GhostCollider {
            grid_x: 6,
            grid_y: 5,
            frightened: false,
        }];

        let events = check_collisions(&pac, &mut pellets, &ghosts);

        assert!(events.is_empty());
    }

    #[test]
    fn multiple_ghosts_on_same_tile() {
        let maze = collision_maze();
        let pac = PacMan::new(5, 5);
        let mut pellets = PelletManager::from_maze(&maze);
        let ghosts = [
            GhostCollider {
                grid_x: 5,
                grid_y: 5,
                frightened: false,
            },
            GhostCollider {
                grid_x: 5,
                grid_y: 5,
                frightened: true,
            },
        ];

        let events = check_collisions(&pac, &mut pellets, &ghosts);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0], CollisionEvent::GhostHit { ghost_index: 0 });
        assert_eq!(events[1], CollisionEvent::GhostEaten { ghost_index: 1 });
    }

    #[test]
    fn no_ghost_events_with_empty_slice() {
        let maze = collision_maze();
        let pac = PacMan::new(5, 5);
        let mut pellets = PelletManager::from_maze(&maze);

        let events = check_collisions(&pac, &mut pellets, &[]);

        // No pellet at (5,5), no ghosts — fully empty.
        assert!(events.is_empty());
    }

    // ── Combined collisions ────────────────────────────────────

    #[test]
    fn pellet_and_ghost_on_same_frame() {
        let maze = collision_maze();
        let pac = PacMan::new(1, 1); // pellet at (1,1)
        let mut pellets = PelletManager::from_maze(&maze);
        let ghosts = [GhostCollider {
            grid_x: 1,
            grid_y: 1,
            frightened: false,
        }];

        let events = check_collisions(&pac, &mut pellets, &ghosts);

        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            CollisionEvent::PelletCollected { x: 1, y: 1 }
        );
        assert_eq!(events[1], CollisionEvent::GhostHit { ghost_index: 0 });
    }

    #[test]
    fn power_pellet_and_frightened_ghost() {
        let maze = collision_maze();
        let pac = PacMan::new(3, 1); // power pellet at (3,1)
        let mut pellets = PelletManager::from_maze(&maze);
        let ghosts = [GhostCollider {
            grid_x: 3,
            grid_y: 1,
            frightened: true,
        }];

        let events = check_collisions(&pac, &mut pellets, &ghosts);

        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            CollisionEvent::PowerPelletCollected { x: 3, y: 1 }
        );
        assert_eq!(events[1], CollisionEvent::GhostEaten { ghost_index: 0 });
    }

    // ── Ghost collider construction ────────────────────────────

    #[test]
    fn ghost_collider_equality() {
        let a = GhostCollider {
            grid_x: 1,
            grid_y: 2,
            frightened: false,
        };
        let b = GhostCollider {
            grid_x: 1,
            grid_y: 2,
            frightened: false,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn ghost_collider_differs_by_position() {
        let a = GhostCollider {
            grid_x: 1,
            grid_y: 2,
            frightened: false,
        };
        let b = GhostCollider {
            grid_x: 2,
            grid_y: 2,
            frightened: false,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn ghost_collider_differs_by_frightened() {
        let a = GhostCollider {
            grid_x: 1,
            grid_y: 2,
            frightened: false,
        };
        let b = GhostCollider {
            grid_x: 1,
            grid_y: 2,
            frightened: true,
        };
        assert_ne!(a, b);
    }

    // ── Integration: movement then collision ───────────────────

    #[test]
    fn moving_pacman_collects_pellets_along_path() {
        let maze = collision_maze();
        let mut pac = PacMan::new(1, 1);
        let mut pellets = PelletManager::from_maze(&maze);
        let dt = 1.0 / 60.0;
        let mut total_events = Vec::new();

        // Check collision at the starting tile before movement begins.
        total_events.extend(check_collisions(&pac, &mut pellets, &[]));

        pac.set_direction(crate::pacman::Direction::Right);

        // Run enough ticks to cross several tiles.
        for _ in 0..100 {
            pac.update(dt, &maze);
            let events = check_collisions(&pac, &mut pellets, &[]);
            total_events.extend(events);
        }

        // Should have collected pellets at (1,1), (2,1), (3,1) along the path.
        let pellet_events: Vec<_> = total_events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    CollisionEvent::PelletCollected { .. }
                        | CollisionEvent::PowerPelletCollected { .. }
                )
            })
            .collect();
        assert_eq!(pellet_events.len(), 3);
    }
}
