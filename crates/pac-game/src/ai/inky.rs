//! Inky (cyan ghost) AI — fickle flanking using Blinky's position.
//!
//! In Chase mode, Inky's target tile is computed relative to both Pac-Man and
//! Blinky. Take the tile 2 spaces ahead of Pac-Man's facing direction, draw a
//! vector from Blinky's position to that tile, then double it. The result is
//! Inky's target. This creates unpredictable "flanking" behavior that depends
//! on Blinky's position — hence the nickname "fickle".
//!
//! In Scatter mode, Inky targets the bottom-right corner of the maze.

use crate::ghost::{Ghost, GhostMode};
use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};
use crate::pacman::{Direction, PacMan};

/// Inky's scatter target — bottom-right corner.
const SCATTER_TARGET: (usize, usize) = (MAZE_WIDTH - 3, MAZE_HEIGHT - 1);

/// Number of tiles ahead of Pac-Man used for the intermediate point.
const LOOK_AHEAD: i32 = 2;

/// Update Inky's requested direction based on the current game state.
///
/// `blinky` is needed because Inky's chase target depends on Blinky's position.
/// Should be called each tick before [`Ghost::update`].
pub fn update(ghost: &mut Ghost, pacman: &PacMan, blinky: &Ghost, maze: &MazeData) {
    if ghost.move_progress() < 1.0 {
        return;
    }

    let target = target_tile(ghost, pacman, blinky);
    let best = pick_direction(ghost, maze, target);

    if let Some(dir) = best {
        ghost.set_direction(dir);
    }
}

/// Compute Inky's target tile based on the current ghost mode.
fn target_tile(ghost: &Ghost, pacman: &PacMan, blinky: &Ghost) -> (usize, usize) {
    match ghost.mode() {
        GhostMode::Chase => chase_target(pacman, blinky),
        GhostMode::Scatter => SCATTER_TARGET,
        GhostMode::Frightened => SCATTER_TARGET,
        GhostMode::Eaten => (ghost.home_x(), ghost.home_y()),
    }
}

/// Chase target: double the vector from Blinky to 2-tiles-ahead-of-Pac-Man.
///
/// 1. Compute the intermediate tile: Pac-Man's position + 2 tiles in his
///    facing direction.
/// 2. Compute the vector from Blinky's tile to the intermediate tile.
/// 3. Double that vector from the intermediate tile.
/// 4. Clamp the result to maze bounds.
///
/// If Pac-Man is stationary (no direction), the intermediate tile is Pac-Man's
/// current position.
fn chase_target(pacman: &PacMan, blinky: &Ghost) -> (usize, usize) {
    let px = pacman.grid_x() as i32;
    let py = pacman.grid_y() as i32;

    let (dx, dy) = pacman
        .current_dir()
        .map(|d| d.delta())
        .unwrap_or((0, 0));

    // Intermediate point: 2 tiles ahead of Pac-Man.
    let ix = px + dx * LOOK_AHEAD;
    let iy = py + dy * LOOK_AHEAD;

    // Vector from Blinky to intermediate point, then double it.
    let bx = blinky.grid_x() as i32;
    let by = blinky.grid_y() as i32;
    let tx = ix + (ix - bx);
    let ty = iy + (iy - by);

    let clamped_x = tx.clamp(0, MAZE_WIDTH as i32 - 1) as usize;
    let clamped_y = ty.clamp(0, MAZE_HEIGHT as i32 - 1) as usize;

    (clamped_x, clamped_y)
}

/// Pick the best direction at the current tile.
///
/// Classic Pac-Man ghost AI: evaluate each passable non-reverse direction and
/// choose the one whose neighbour tile is closest (Euclidean) to the target.
/// Ties are broken by priority: Up > Left > Down > Right.
fn pick_direction(
    ghost: &Ghost,
    maze: &MazeData,
    target: (usize, usize),
) -> Option<Direction> {
    let gx = ghost.grid_x();
    let gy = ghost.grid_y();
    let reverse = ghost.current_dir().map(|d| d.opposite());

    const PRIORITY: [Direction; 4] = [
        Direction::Up,
        Direction::Left,
        Direction::Down,
        Direction::Right,
    ];

    let mut best_dir: Option<Direction> = None;
    let mut best_dist = f64::MAX;

    for &dir in &PRIORITY {
        if Some(dir) == reverse {
            continue;
        }

        let (nx, ny) = neighbour(gx, gy, dir);

        if !is_passable_for_ghost(ghost, maze, nx, ny) {
            continue;
        }

        let dist = euclidean_sq(nx, ny, target.0, target.1);
        if dist < best_dist {
            best_dist = dist;
            best_dir = Some(dir);
        }
    }

    // Dead end: allow reverse.
    if best_dir.is_none() {
        if let Some(rev) = reverse {
            let (nx, ny) = neighbour(gx, gy, rev);
            if is_passable_for_ghost(ghost, maze, nx, ny) {
                best_dir = Some(rev);
            }
        }
    }

    best_dir
}

/// Squared Euclidean distance between two tile positions.
fn euclidean_sq(ax: usize, ay: usize, bx: usize, by: usize) -> f64 {
    let dx = ax as f64 - bx as f64;
    let dy = ay as f64 - by as f64;
    dx * dx + dy * dy
}

/// Compute the neighbour tile in the given direction, with tunnel wrapping.
fn neighbour(x: usize, y: usize, dir: Direction) -> (usize, usize) {
    let (dx, dy) = dir.delta();
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;

    let wrapped_x = if nx < 0 {
        MAZE_WIDTH - 1
    } else if nx >= MAZE_WIDTH as i32 {
        0
    } else {
        nx as usize
    };

    let wrapped_y = ny.clamp(0, MAZE_HEIGHT as i32 - 1) as usize;

    (wrapped_x, wrapped_y)
}

/// Check whether a tile is passable for this ghost.
fn is_passable_for_ghost(ghost: &Ghost, maze: &MazeData, x: usize, y: usize) -> bool {
    match maze.get(x, y) {
        Some(TileType::Wall) => false,
        Some(TileType::GhostHouse) | Some(TileType::GhostDoor) => {
            ghost.in_ghost_house() || ghost.mode() == GhostMode::Eaten
        }
        Some(_) => true,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ghost::GhostId;

    /// Open maze with border walls.
    fn test_maze() -> MazeData {
        let mut tiles = [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT];
        for x in 0..MAZE_WIDTH {
            tiles[0][x] = TileType::Wall;
            tiles[MAZE_HEIGHT - 1][x] = TileType::Wall;
        }
        for y in 0..MAZE_HEIGHT {
            tiles[y][0] = TileType::Wall;
            tiles[y][MAZE_WIDTH - 1] = TileType::Wall;
        }
        MazeData { tiles }
    }

    // ── Chase targeting ───────────────────────────────────────

    #[test]
    fn chase_target_doubles_vector_from_blinky() {
        // Pac-Man at (10, 15) stationary, Blinky at (10, 10).
        // Intermediate = (10, 15) (stationary, no look-ahead offset).
        // Vector from Blinky: (0, 5). Double: target = (10, 20).
        let pac = PacMan::new(10, 15);
        let blinky = Ghost::new(GhostId::Blinky, 10, 10);
        let (tx, ty) = chase_target(&pac, &blinky);
        assert_eq!((tx, ty), (10, 20));
    }

    #[test]
    fn chase_target_with_blinky_behind_pacman() {
        // Pac-Man at (14, 15) stationary, Blinky at (20, 15).
        // Intermediate = (14, 15). Vector from Blinky: (-6, 0). Double: target = (8, 15).
        let pac = PacMan::new(14, 15);
        let blinky = Ghost::new(GhostId::Blinky, 20, 15);
        let (tx, ty) = chase_target(&pac, &blinky);
        assert_eq!((tx, ty), (8, 15));
    }

    #[test]
    fn chase_target_clamps_to_maze_bounds() {
        // Pac-Man at (2, 2) stationary, Blinky at (20, 20).
        // Intermediate = (2, 2). Vector: (-18, -18). Target: (-16, -16) → clamped to (0, 0).
        let pac = PacMan::new(2, 2);
        let blinky = Ghost::new(GhostId::Blinky, 20, 20);
        let (tx, ty) = chase_target(&pac, &blinky);
        assert_eq!(tx, 0);
        assert_eq!(ty, 0);
    }

    #[test]
    fn chase_target_when_blinky_at_intermediate() {
        // Blinky right on top of the intermediate point → vector is (0,0),
        // target equals intermediate equals Pac-Man's tile (stationary).
        let pac = PacMan::new(10, 15);
        let blinky = Ghost::new(GhostId::Blinky, 10, 15);
        let (tx, ty) = chase_target(&pac, &blinky);
        assert_eq!((tx, ty), (10, 15));
    }

    // ── Scatter targeting ─────────────────────────────────────

    #[test]
    fn scatter_target_is_bottom_right() {
        let ghost = Ghost::new(GhostId::Inky, 14, 14);
        let pac = PacMan::new(5, 5);
        let blinky = Ghost::new(GhostId::Blinky, 10, 10);
        let target = target_tile(&ghost, &pac, &blinky);
        assert_eq!(target, SCATTER_TARGET);
    }

    // ── Eaten targeting ───────────────────────────────────────

    #[test]
    fn eaten_targets_home_tile() {
        let mut ghost = Ghost::new(GhostId::Inky, 14, 14);
        ghost.set_mode(GhostMode::Eaten);
        let pac = PacMan::new(5, 5);
        let blinky = Ghost::new(GhostId::Blinky, 10, 10);
        let target = target_tile(&ghost, &pac, &blinky);
        assert_eq!(target, (14, 14));
    }

    // ── Direction picking ─────────────────────────────────────

    #[test]
    fn pick_direction_toward_target() {
        let maze = test_maze();
        let ghost = Ghost::new(GhostId::Inky, 5, 5);
        // Target below and right.
        let dir = pick_direction(&ghost, &maze, (20, 20));
        assert!(dir.is_some());
        let d = dir.unwrap();
        assert!(
            d == Direction::Down || d == Direction::Right,
            "should move toward target, got {:?}",
            d
        );
    }

    #[test]
    fn pick_direction_avoids_walls() {
        let mut maze = test_maze();
        maze.tiles[5][6] = TileType::Wall;
        let ghost = Ghost::new(GhostId::Inky, 5, 5);
        let dir = pick_direction(&ghost, &maze, (20, 5));
        assert!(dir.is_some());
        assert_ne!(dir.unwrap(), Direction::Right);
    }

    // ── Full update integration ───────────────────────────────

    #[test]
    fn update_sets_direction_in_chase() {
        let maze = test_maze();
        let mut ghost = Ghost::new(GhostId::Inky, 10, 10);
        ghost.set_mode(GhostMode::Chase);
        let pac = PacMan::new(20, 10);
        // Blinky at (5, 10) → intermediate (20,10), vector (15, 0), target (35,10)→clamped right.
        let blinky = Ghost::new(GhostId::Blinky, 5, 10);

        update(&mut ghost, &pac, &blinky, &maze);

        assert!(ghost.requested_dir().is_some());
        assert_eq!(ghost.requested_dir(), Some(Direction::Right));
    }

    #[test]
    fn update_skips_when_mid_tile() {
        let maze = test_maze();
        let mut ghost = Ghost::new(GhostId::Inky, 10, 10);
        ghost.set_mode(GhostMode::Chase);
        let pac = PacMan::new(20, 10);
        let blinky = Ghost::new(GhostId::Blinky, 5, 10);

        // Start movement to create mid-tile state.
        ghost.set_direction(Direction::Right);
        ghost.update(1.0 / 60.0, &maze);
        assert!(ghost.move_progress() < 1.0);

        // Clear queued dir, then call update — should not set a new direction.
        let prev_queued = ghost.queued_dir();
        update(&mut ghost, &pac, &blinky, &maze);
        assert_eq!(ghost.queued_dir(), prev_queued);
    }

    // ── Neighbour wrapping ────────────────────────────────────

    #[test]
    fn neighbour_wraps_tunnel() {
        let (nx, _) = neighbour(0, 15, Direction::Left);
        assert_eq!(nx, MAZE_WIDTH - 1);

        let (nx, _) = neighbour(MAZE_WIDTH - 1, 15, Direction::Right);
        assert_eq!(nx, 0);
    }

    // ── Fickle behavior demonstration ─────────────────────────

    #[test]
    fn target_changes_when_blinky_moves() {
        // Same Pac-Man position, different Blinky positions → different targets.
        let pac = PacMan::new(14, 15);
        let blinky_a = Ghost::new(GhostId::Blinky, 10, 15);
        let blinky_b = Ghost::new(GhostId::Blinky, 18, 15);

        let target_a = chase_target(&pac, &blinky_a);
        let target_b = chase_target(&pac, &blinky_b);

        assert_ne!(target_a, target_b, "Inky target should change with Blinky position");
        // blinky_a at (10, 15): vector (4, 0), target (18, 15)
        assert_eq!(target_a, (18, 15));
        // blinky_b at (18, 15): vector (-4, 0), target (10, 15)
        assert_eq!(target_b, (10, 15));
    }
}
