//! Blinky (Red Ghost) AI — direct chase with Cruise Elroy speedup.
//!
//! Blinky is the most aggressive ghost. In chase mode he targets Pac-Man's
//! current tile directly. When the remaining pellet count drops below
//! thresholds, Blinky enters "Cruise Elroy" mode with increased speed and
//! continues chasing even during scatter phases.

use crate::ghost::{Ghost, GhostMode};
use crate::maze::MazeData;
use crate::pacman::{Direction, PacMan};

/// Blinky's scatter target: top-right area of the maze.
const SCATTER_TARGET: (usize, usize) = (25, 0);

/// Tile just above the ghost house door (eaten-ghost pathfinding target).
const GHOST_HOUSE_ENTRANCE: (usize, usize) = (13, 10);

/// Pellets remaining threshold for Cruise Elroy level 1.
const ELROY1_THRESHOLD: usize = 20;
/// Pellets remaining threshold for Cruise Elroy level 2.
const ELROY2_THRESHOLD: usize = 10;

/// Cruise Elroy level 1 speed in tiles/sec (normal ghost speed is 7.5).
const ELROY1_SPEED: f32 = 8.0;
/// Cruise Elroy level 2 speed in tiles/sec.
const ELROY2_SPEED: f32 = 8.5;

/// Direction evaluation order — defines tie-breaking priority.
///
/// Classic Pac-Man breaks ties as Up > Left > Down > Right.
const DIRECTION_PRIORITY: [Direction; 4] = [
    Direction::Up,
    Direction::Left,
    Direction::Down,
    Direction::Right,
];

/// Blinky (Red Ghost) AI controller.
///
/// Call [`update`](BlinkyAi::update) each tick to steer the ghost. The ghost
/// entity handles movement mechanics; this controller decides direction and
/// manages Cruise Elroy speed.
pub struct BlinkyAi {
    /// Current Cruise Elroy level: 0 = off, 1 = level 1, 2 = level 2.
    elroy_level: u8,
    /// Monotonic counter for pseudo-random frightened direction selection.
    tick: u32,
}

impl BlinkyAi {
    pub fn new() -> Self {
        Self {
            elroy_level: 0,
            tick: 0,
        }
    }

    /// Current Cruise Elroy level (0 = off, 1 = faster, 2 = fastest).
    pub fn elroy_level(&self) -> u8 {
        self.elroy_level
    }

    /// Update Blinky for one tick.
    ///
    /// Sets direction when the ghost reaches a tile boundary and manages
    /// Cruise Elroy speed based on remaining pellets.
    pub fn update(
        &mut self,
        ghost: &mut Ghost,
        pacman: &PacMan,
        maze: &MazeData,
        remaining_pellets: usize,
    ) {
        self.tick = self.tick.wrapping_add(1);
        self.update_elroy(ghost, remaining_pellets);

        let dir = match ghost.mode() {
            GhostMode::Frightened => frightened_direction(ghost, maze, self.tick),
            _ => {
                let target = self.target_tile(ghost, pacman);
                best_direction_toward(ghost, target, maze)
            }
        };

        if let Some(d) = dir {
            ghost.set_direction(d);
        }
    }

    /// Determine the target tile based on current mode and Elroy status.
    fn target_tile(&self, ghost: &Ghost, pacman: &PacMan) -> (usize, usize) {
        match ghost.mode() {
            GhostMode::Chase => (pacman.grid_x(), pacman.grid_y()),
            GhostMode::Scatter => {
                if self.elroy_level > 0 {
                    // Cruise Elroy: ignore scatter, keep chasing.
                    (pacman.grid_x(), pacman.grid_y())
                } else {
                    SCATTER_TARGET
                }
            }
            GhostMode::Eaten => GHOST_HOUSE_ENTRANCE,
            GhostMode::Frightened => SCATTER_TARGET, // unused — frightened picks randomly
        }
    }

    /// Update Cruise Elroy level and apply/clear speed override.
    fn update_elroy(&mut self, ghost: &mut Ghost, remaining_pellets: usize) {
        self.elroy_level = if remaining_pellets <= ELROY2_THRESHOLD {
            2
        } else if remaining_pellets <= ELROY1_THRESHOLD {
            1
        } else {
            0
        };

        // Speed override only applies during Chase/Scatter.
        match ghost.mode() {
            GhostMode::Chase | GhostMode::Scatter => match self.elroy_level {
                2 => ghost.set_speed_override(ELROY2_SPEED),
                1 => ghost.set_speed_override(ELROY1_SPEED),
                _ => ghost.clear_speed_override(),
            },
            _ => ghost.clear_speed_override(),
        }
    }
}

/// Pick the direction that minimises distance to `target`.
///
/// Classic Pac-Man ghost AI algorithm:
/// 1. Exclude the reverse of the current direction (no-reverse rule).
/// 2. For each passable direction, compute squared Euclidean distance from
///    the resulting neighbor tile to `target`.
/// 3. Pick the direction with minimum distance.
/// 4. Tie-break with priority: Up > Left > Down > Right.
fn best_direction_toward(
    ghost: &Ghost,
    target: (usize, usize),
    maze: &MazeData,
) -> Option<Direction> {
    let reverse = ghost.current_dir().map(|d| d.opposite());

    let mut best_dir = None;
    let mut best_dist = u64::MAX;

    for &dir in &DIRECTION_PRIORITY {
        if Some(dir) == reverse {
            continue;
        }
        if !ghost.is_direction_passable(dir, maze) {
            continue;
        }

        let (nx, ny) = ghost.neighbor_tile(dir);
        let dist = distance_sq(nx, ny, target.0, target.1);

        if dist < best_dist {
            best_dist = dist;
            best_dir = Some(dir);
        }
    }

    best_dir
}

/// Pick a pseudo-random valid direction for frightened mode.
///
/// Uses a hash of position and tick counter to select among available
/// non-reverse directions. Deterministic but varied-looking behavior.
fn frightened_direction(ghost: &Ghost, maze: &MazeData, tick: u32) -> Option<Direction> {
    let reverse = ghost.current_dir().map(|d| d.opposite());

    let mut candidates = Vec::new();
    for &dir in &DIRECTION_PRIORITY {
        if Some(dir) == reverse {
            continue;
        }
        if ghost.is_direction_passable(dir, maze) {
            candidates.push(dir);
        }
    }

    if candidates.is_empty() {
        // Dead end: allow reverse.
        if let Some(rev) = reverse {
            if ghost.is_direction_passable(rev, maze) {
                return Some(rev);
            }
        }
        return None;
    }

    let hash = (ghost.grid_x() as u32)
        .wrapping_mul(31)
        .wrapping_add((ghost.grid_y() as u32).wrapping_mul(17))
        .wrapping_add(tick.wrapping_mul(7));
    let idx = hash as usize % candidates.len();
    Some(candidates[idx])
}

/// Squared Euclidean distance between two grid positions.
fn distance_sq(x1: usize, y1: usize, x2: usize, y2: usize) -> u64 {
    let dx = x1 as i64 - x2 as i64;
    let dy = y1 as i64 - y2 as i64;
    (dx * dx + dy * dy) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};

    const DT: f32 = 1.0 / 60.0;

    /// Open maze with border walls — standard test arena.
    fn open_maze() -> MazeData {
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

    /// Corridor maze: horizontal corridor at row 5 with walls above and below.
    fn corridor_maze() -> MazeData {
        let mut tiles = [[TileType::Wall; MAZE_WIDTH]; MAZE_HEIGHT];
        for x in 1..MAZE_WIDTH - 1 {
            tiles[5][x] = TileType::Empty;
        }
        MazeData { tiles }
    }

    /// T-junction maze: horizontal corridor at row 5, vertical corridor at col 10.
    fn t_junction_maze() -> MazeData {
        let mut tiles = [[TileType::Wall; MAZE_WIDTH]; MAZE_HEIGHT];
        for x in 1..MAZE_WIDTH - 1 {
            tiles[5][x] = TileType::Empty;
        }
        for y in 1..MAZE_HEIGHT - 1 {
            tiles[y][10] = TileType::Empty;
        }
        MazeData { tiles }
    }

    /// Helper: start a ghost moving so it has a real current_dir.
    fn start_moving(ghost: &mut Ghost, dir: Direction, maze: &MazeData) {
        ghost.set_direction(dir);
        ghost.update(DT, maze);
    }

    // ── Construction ──────────────────────────────────────────

    #[test]
    fn new_starts_at_elroy_zero() {
        let ai = BlinkyAi::new();
        assert_eq!(ai.elroy_level(), 0);
    }

    // ── Distance ──────────────────────────────────────────────

    #[test]
    fn distance_sq_same_point() {
        assert_eq!(distance_sq(5, 5, 5, 5), 0);
    }

    #[test]
    fn distance_sq_horizontal() {
        assert_eq!(distance_sq(0, 0, 3, 0), 9);
    }

    #[test]
    fn distance_sq_diagonal() {
        assert_eq!(distance_sq(0, 0, 3, 4), 25);
    }

    // ── Chase targeting ───────────────────────────────────────

    #[test]
    fn chase_targets_pacman_directly() {
        let maze = open_maze();
        // Ghost at (5, 5), Pac-Man at (5, 10) — directly below.
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        let pacman = PacMan::new(5, 10);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);

        // Should pick Down (toward Pac-Man who is below).
        assert_eq!(ghost.requested_dir(), Some(Direction::Down));
    }

    #[test]
    fn chase_picks_shortest_direction() {
        let maze = open_maze();
        // Ghost at (10, 5), Pac-Man at (20, 5) — to the right.
        let mut ghost = Ghost::new(10, 5);
        ghost.set_mode(GhostMode::Chase);
        let pacman = PacMan::new(20, 5);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);

        assert_eq!(ghost.requested_dir(), Some(Direction::Right));
    }

    #[test]
    fn chase_picks_left_when_pacman_is_left() {
        let maze = open_maze();
        let mut ghost = Ghost::new(15, 5);
        ghost.set_mode(GhostMode::Chase);
        let pacman = PacMan::new(5, 5);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);

        assert_eq!(ghost.requested_dir(), Some(Direction::Left));
    }

    // ── Scatter targeting ─────────────────────────────────────

    #[test]
    fn scatter_targets_corner() {
        let maze = open_maze();
        // Ghost at (10, 15) in scatter mode — should move toward top-right (25, 0).
        let mut ghost = Ghost::new(10, 15);
        // Ghost starts in Scatter, so no mode change needed.
        let pacman = PacMan::new(5, 20);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);

        // Target is (25, 0): up and right. Up has higher priority in ties,
        // but Right reduces more distance. Let's verify the direction is sensible.
        let dir = ghost.requested_dir().unwrap();
        // (10,15) → (25,0): dx=15, dy=-15. Right tile (11,15): dist=196+225=421.
        // Up tile (10,14): dist=225+196=421. Tie → Up wins by priority.
        assert_eq!(dir, Direction::Up);
    }

    // ── Cruise Elroy ──────────────────────────────────────────

    #[test]
    fn elroy_level1_at_20_pellets() {
        let maze = open_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        let pacman = PacMan::new(10, 10);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 20);
        assert_eq!(ai.elroy_level(), 1);
    }

    #[test]
    fn elroy_level2_at_10_pellets() {
        let maze = open_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        let pacman = PacMan::new(10, 10);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 10);
        assert_eq!(ai.elroy_level(), 2);
    }

    #[test]
    fn elroy_off_above_threshold() {
        let maze = open_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        let pacman = PacMan::new(10, 10);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 21);
        assert_eq!(ai.elroy_level(), 0);
    }

    #[test]
    fn elroy1_speed_override_applied() {
        let maze = open_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        let pacman = PacMan::new(10, 10);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 20);
        assert!((ghost.speed() - ELROY1_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn elroy2_speed_override_applied() {
        let maze = open_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        let pacman = PacMan::new(10, 10);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 10);
        assert!((ghost.speed() - ELROY2_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn elroy_speed_cleared_in_frightened() {
        let maze = open_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Frightened);
        let pacman = PacMan::new(10, 10);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 5);
        // Elroy level set, but speed override cleared for frightened mode.
        assert_eq!(ai.elroy_level(), 2);
        assert!((ghost.speed() - 5.0).abs() < f32::EPSILON); // FRIGHTENED_SPEED
    }

    #[test]
    fn elroy_scatter_targets_pacman() {
        let maze = open_maze();
        // Ghost in scatter mode with Elroy active should target Pac-Man, not corner.
        let mut ghost = Ghost::new(10, 5);
        // Starts in Scatter mode (default).
        let pacman = PacMan::new(20, 5);
        let mut ai = BlinkyAi::new();

        // Activate Elroy.
        ai.update(&mut ghost, &pacman, &maze, 15);
        assert_eq!(ai.elroy_level(), 1);

        // Should target Pac-Man (right), not scatter corner.
        assert_eq!(ghost.requested_dir(), Some(Direction::Right));
    }

    // ── No-reverse rule ───────────────────────────────────────

    #[test]
    fn does_not_reverse_in_corridor() {
        let maze = corridor_maze();
        // Ghost moving right in a horizontal corridor.
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        start_moving(&mut ghost, Direction::Right, &maze);
        // Now current_dir=Right, heading toward (6, 5).

        let pacman = PacMan::new(2, 5); // Pac-Man is to the left.
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);

        // In a corridor, Up/Down are walls, Left is reverse → only Right.
        assert_eq!(ghost.requested_dir(), Some(Direction::Right));
    }

    // ── Intersection behavior ─────────────────────────────────

    #[test]
    fn picks_best_at_intersection() {
        let maze = t_junction_maze();
        // Ghost approaching the T-junction from the left with current_dir=Right.
        let mut ghost = Ghost::new(9, 5);
        ghost.set_mode(GhostMode::Chase);
        start_moving(&mut ghost, Direction::Right, &maze);
        // Now heading toward (10, 5), current_dir=Right.

        let pacman = PacMan::new(10, 1);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);
        // From (10, 5) toward (10, 1): Up (10,4) dist=9, Down (10,6) dist=25,
        // Right (11,5) dist=17. Left is reverse → skip. Up wins.
        assert_eq!(ghost.requested_dir(), Some(Direction::Up));
    }

    // ── Frightened mode ───────────────────────────────────────

    #[test]
    fn frightened_picks_valid_direction() {
        let maze = open_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Frightened);
        let pacman = PacMan::new(10, 10);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);

        let dir = ghost.requested_dir();
        assert!(dir.is_some(), "frightened ghost should pick a direction");
        // Direction should be passable.
        assert!(ghost.is_direction_passable(dir.unwrap(), &maze));
    }

    #[test]
    fn frightened_does_not_reverse_when_options_exist() {
        let maze = t_junction_maze();
        // Ghost approaching intersection with current_dir=Right.
        let mut ghost = Ghost::new(9, 5);
        ghost.set_mode(GhostMode::Frightened);
        start_moving(&mut ghost, Direction::Right, &maze);
        // Now heading toward (10, 5), current_dir=Right.

        let pacman = PacMan::new(1, 1);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);

        let dir = ghost.requested_dir().unwrap();
        // At (10, 5): Up, Down, Right available; Left is reverse → excluded.
        assert_ne!(dir, Direction::Left);
    }

    // ── Eaten mode ────────────────────────────────────────────

    #[test]
    fn eaten_targets_ghost_house_entrance() {
        let maze = open_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Eaten);
        let pacman = PacMan::new(20, 20);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);

        let dir = ghost.requested_dir().unwrap();
        // Ghost house entrance is (13, 10). From (5, 5): Right and Down reduce distance.
        // Right tile (6, 5): dist=(6-13)^2+(5-10)^2 = 49+25=74
        // Down tile (5, 6): dist=(5-13)^2+(6-10)^2 = 64+16=80
        // Up tile (5, 4): dist=(5-13)^2+(4-10)^2 = 64+36=100
        // Should pick Right (minimum distance).
        assert_eq!(dir, Direction::Right);
    }

    // ── Direction priority tie-breaking ───────────────────────

    #[test]
    fn tie_break_favors_up_over_left() {
        // When Up and Left yield equal distance, Up should win.
        let maze = open_maze();
        // Ghost at (10, 10), target at (7, 7) — equal dx and dy.
        // Up tile (10, 9): dist=(10-7)^2+(9-7)^2 = 9+4=13
        // Left tile (9, 10): dist=(9-7)^2+(10-7)^2 = 4+9=13
        // Tie → Up wins.
        let mut ghost = Ghost::new(10, 10);
        ghost.set_mode(GhostMode::Chase);
        let pacman = PacMan::new(7, 7);
        let mut ai = BlinkyAi::new();

        ai.update(&mut ghost, &pacman, &maze, 100);
        assert_eq!(ghost.requested_dir(), Some(Direction::Up));
    }

    // ── Ghost speed override on Ghost struct ──────────────────

    #[test]
    fn ghost_speed_override() {
        let mut ghost = Ghost::new(5, 5);
        let base_speed = ghost.speed();
        ghost.set_speed_override(10.0);
        assert!((ghost.speed() - 10.0).abs() < f32::EPSILON);
        ghost.clear_speed_override();
        assert!((ghost.speed() - base_speed).abs() < f32::EPSILON);
    }
}
