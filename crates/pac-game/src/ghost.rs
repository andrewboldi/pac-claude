//! Ghost base entity — state machine with Chase/Scatter/Frightened/Eaten modes,
//! grid movement, and smooth interpolation.
//!
//! Direction selection is handled externally by AI systems (Blinky, Pinky, etc.)
//! via [`Ghost::set_direction`]. This module provides movement mechanics,
//! ghost-house passability rules, and mode transitions with direction reversal.

use crate::collision::GhostCollider;
use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};
use crate::pacman::Direction;

/// Movement speed in tiles per second (Chase / Scatter).
const NORMAL_SPEED: f32 = 7.5;
/// Movement speed in tiles per second (Frightened).
const FRIGHTENED_SPEED: f32 = 5.0;
/// Movement speed in tiles per second (Eaten — eyes returning home).
const EATEN_SPEED: f32 = 15.0;

/// Ghost behavioral mode (state machine).
///
/// Transitions are triggered externally by the mode controller. The ghost
/// entity enforces side effects: speed changes and direction reversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostMode {
    /// Actively chasing Pac-Man using ghost-specific targeting.
    Chase,
    /// Returning to home corner using a fixed scatter target.
    Scatter,
    /// Vulnerable — can be eaten by Pac-Man. Moves slowly.
    Frightened,
    /// Eyes only — returning to ghost house after being eaten.
    Eaten,
}

impl GhostMode {
    /// Movement speed for this mode in tiles per second.
    pub fn speed(self) -> f32 {
        match self {
            GhostMode::Chase | GhostMode::Scatter => NORMAL_SPEED,
            GhostMode::Frightened => FRIGHTENED_SPEED,
            GhostMode::Eaten => EATEN_SPEED,
        }
    }
}

/// Ghost entity with grid-based movement and a behavioral state machine.
///
/// Ghosts spawn in or near the ghost house and are steered by an external AI
/// system that calls [`set_direction`](Ghost::set_direction) each tick. The
/// ghost handles movement, wall collision, tunnel wrapping, and ghost-house
/// passability.
///
/// The no-reverse rule is enforced: ghosts cannot reverse direction mid-tile
/// unless triggered by a mode change via [`set_mode`](Ghost::set_mode).
pub struct Ghost {
    /// Current grid column.
    grid_x: usize,
    /// Current grid row.
    grid_y: usize,
    /// Grid column moving from (for interpolation).
    prev_x: usize,
    /// Grid row moving from (for interpolation).
    prev_y: usize,
    /// Direction currently being travelled.
    current_dir: Option<Direction>,
    /// Direction requested by AI, applied at next tile boundary.
    requested_dir: Option<Direction>,
    /// Progress from prev toward current tile, 0.0–1.0.
    move_progress: f32,
    /// Current behavioral mode.
    mode: GhostMode,
    /// Ghost house home column (where eaten ghosts return).
    home_x: usize,
    /// Ghost house home row (where eaten ghosts return).
    home_y: usize,
    /// Whether this ghost is currently inside the ghost house.
    in_ghost_house: bool,
}

impl Ghost {
    /// Create a ghost at the given grid position, outside the ghost house.
    ///
    /// The home tile defaults to the spawn position. Mode starts as Scatter
    /// (classic Pac-Man ghosts begin in scatter mode).
    pub fn new(grid_x: usize, grid_y: usize) -> Self {
        Self {
            grid_x,
            grid_y,
            prev_x: grid_x,
            prev_y: grid_y,
            current_dir: None,
            requested_dir: None,
            move_progress: 1.0,
            mode: GhostMode::Scatter,
            home_x: grid_x,
            home_y: grid_y,
            in_ghost_house: false,
        }
    }

    /// Create a ghost spawned inside the ghost house.
    ///
    /// Identical to [`new`](Ghost::new) but with `in_ghost_house` set,
    /// allowing the ghost to traverse ghost-house tiles and the ghost door.
    pub fn new_in_house(grid_x: usize, grid_y: usize) -> Self {
        let mut ghost = Self::new(grid_x, grid_y);
        ghost.in_ghost_house = true;
        ghost
    }

    // ── Accessors ─────────────────────────────────────────────

    pub fn grid_x(&self) -> usize {
        self.grid_x
    }

    pub fn grid_y(&self) -> usize {
        self.grid_y
    }

    pub fn prev_x(&self) -> usize {
        self.prev_x
    }

    pub fn prev_y(&self) -> usize {
        self.prev_y
    }

    pub fn current_dir(&self) -> Option<Direction> {
        self.current_dir
    }

    pub fn requested_dir(&self) -> Option<Direction> {
        self.requested_dir
    }

    pub fn move_progress(&self) -> f32 {
        self.move_progress
    }

    pub fn mode(&self) -> GhostMode {
        self.mode
    }

    pub fn home_x(&self) -> usize {
        self.home_x
    }

    pub fn home_y(&self) -> usize {
        self.home_y
    }

    pub fn in_ghost_house(&self) -> bool {
        self.in_ghost_house
    }

    /// Movement speed for the current mode in tiles per second.
    pub fn speed(&self) -> f32 {
        self.mode.speed()
    }

    // ── Direction & mode ──────────────────────────────────────

    /// Request a movement direction from the AI.
    ///
    /// Applied at the next tile boundary. The no-reverse rule is enforced
    /// during [`update`](Ghost::update): a reverse request is ignored unless
    /// the ghost is at a dead end.
    pub fn set_direction(&mut self, dir: Direction) {
        self.requested_dir = Some(dir);
    }

    /// Transition to a new mode.
    ///
    /// Classic Pac-Man reversal rules:
    /// - Chase ↔ Scatter: reverse direction
    /// - Any → Frightened: reverse direction
    /// - Any → Eaten: NO reverse (ghost was just caught)
    /// - Frightened/Eaten → Chase/Scatter: no reverse
    pub fn set_mode(&mut self, new_mode: GhostMode) {
        let old_mode = self.mode;
        if old_mode == new_mode {
            return;
        }

        let should_reverse = matches!(
            (old_mode, new_mode),
            (GhostMode::Chase, GhostMode::Scatter)
                | (GhostMode::Scatter, GhostMode::Chase)
                | (GhostMode::Chase, GhostMode::Frightened)
                | (GhostMode::Scatter, GhostMode::Frightened)
        );

        self.mode = new_mode;

        if should_reverse {
            self.reverse();
        }
    }

    // ── Update ────────────────────────────────────────────────

    /// Advance ghost by one fixed timestep.
    ///
    /// `dt` is the fixed timestep duration in seconds. Movement is checked
    /// against the maze for wall collisions and ghost-house passability.
    /// Tunnel wrapping is applied at horizontal edges.
    pub fn update(&mut self, dt: f32, maze: &MazeData) {
        let speed = self.speed();

        // Advance progress toward target tile.
        if self.current_dir.is_some() && self.move_progress < 1.0 {
            self.move_progress += speed * dt;
        }

        // Arrived at target tile?
        if self.move_progress >= 1.0 {
            self.move_progress = 1.0;

            // Update ghost house tracking based on current tile.
            self.update_house_state(maze);

            // Pick next direction.
            let next_dir = self.pick_next_direction(maze);

            if let Some(dir) = next_dir {
                if self.requested_dir == Some(dir) {
                    self.requested_dir = None;
                }
                self.begin_move(dir, maze);
            } else {
                self.current_dir = None;
            }
        }
    }

    /// Interpolated world-space position for rendering.
    ///
    /// Returns `(world_x, world_z)` matching the maze renderer convention
    /// where grid `(x, y)` maps to world `(x, 0, y)`.
    pub fn world_position(&self, _alpha: f32) -> (f32, f32) {
        let t = self.move_progress.clamp(0.0, 1.0);
        let visual_x = self.prev_x as f32 + (self.grid_x as f32 - self.prev_x as f32) * t;
        let visual_z = self.prev_y as f32 + (self.grid_y as f32 - self.prev_y as f32) * t;
        (visual_x, visual_z)
    }

    /// Convert to a [`GhostCollider`] for the collision system.
    ///
    /// Eaten ghosts are not collidable (eyes only), so the collision system
    /// should skip them. The `frightened` field drives eat-vs-hit logic.
    pub fn to_collider(&self) -> GhostCollider {
        GhostCollider {
            grid_x: self.grid_x,
            grid_y: self.grid_y,
            frightened: self.mode == GhostMode::Frightened,
        }
    }

    /// Whether this eaten ghost has arrived at its home tile.
    ///
    /// The mode controller should transition the ghost back to Chase/Scatter
    /// (via the ghost house) when this returns `true`.
    pub fn reached_home(&self) -> bool {
        self.mode == GhostMode::Eaten
            && self.grid_x == self.home_x
            && self.grid_y == self.home_y
            && self.move_progress >= 1.0
    }

    // ── Internal ──────────────────────────────────────────────

    /// Reverse direction mid-tile or at a boundary.
    fn reverse(&mut self) {
        if let Some(dir) = self.current_dir {
            if self.move_progress < 1.0 {
                // Mid-tile: swap prev and current, invert progress.
                let old_gx = self.grid_x;
                let old_gy = self.grid_y;
                self.grid_x = self.prev_x;
                self.grid_y = self.prev_y;
                self.prev_x = old_gx;
                self.prev_y = old_gy;
                self.move_progress = 1.0 - self.move_progress;
            }
            self.current_dir = Some(dir.opposite());
        }
    }

    /// Pick the next direction at a tile boundary.
    ///
    /// Priority: requested (non-reverse) → current → requested (reverse, dead end only).
    fn pick_next_direction(&self, maze: &MazeData) -> Option<Direction> {
        let reverse_dir = self.current_dir.map(|d| d.opposite());

        // Try requested direction if passable and not a reverse.
        if let Some(req) = self.requested_dir {
            if self.can_move(req, maze) && Some(req) != reverse_dir {
                return Some(req);
            }
        }

        // Continue current direction if passable.
        if let Some(current) = self.current_dir {
            if self.can_move(current, maze) {
                return Some(current);
            }
        }

        // Dead end: allow reverse via requested direction.
        if let Some(req) = self.requested_dir {
            if self.can_move(req, maze) {
                return Some(req);
            }
        }

        // Last resort: try the reverse of current direction.
        if let Some(rev) = reverse_dir {
            if self.can_move(rev, maze) {
                return Some(rev);
            }
        }

        None
    }

    /// Whether the ghost can move in the given direction from the current tile.
    fn can_move(&self, dir: Direction, maze: &MazeData) -> bool {
        let (tx, ty) = self.target_tile(self.grid_x, self.grid_y, dir);
        self.is_passable(maze, tx, ty)
    }

    /// Start moving toward the adjacent tile in `dir`.
    fn begin_move(&mut self, dir: Direction, _maze: &MazeData) {
        let (tx, ty) = self.target_tile(self.grid_x, self.grid_y, dir);
        self.prev_x = self.grid_x;
        self.prev_y = self.grid_y;
        self.grid_x = tx;
        self.grid_y = ty;
        self.current_dir = Some(dir);
        self.move_progress = 0.0;
    }

    /// Compute the target tile from `(x, y)` in direction `dir`, applying
    /// horizontal tunnel wrapping.
    fn target_tile(&self, x: usize, y: usize, dir: Direction) -> (usize, usize) {
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

    /// Ghost-specific passability: walls block, ghost house/door are conditional.
    fn is_passable(&self, maze: &MazeData, x: usize, y: usize) -> bool {
        match maze.get(x, y) {
            Some(TileType::Wall) => false,
            Some(TileType::GhostHouse) => self.in_ghost_house || self.mode == GhostMode::Eaten,
            Some(TileType::GhostDoor) => self.in_ghost_house || self.mode == GhostMode::Eaten,
            Some(_) => true,
            None => false,
        }
    }

    /// Update `in_ghost_house` based on the tile the ghost just arrived at.
    fn update_house_state(&mut self, maze: &MazeData) {
        let tile = maze.get(self.grid_x, self.grid_y);
        match tile {
            Some(TileType::GhostHouse) | Some(TileType::GhostDoor) => {
                // Eaten ghost arriving at home re-enters the house.
                if self.mode == GhostMode::Eaten {
                    self.in_ghost_house = true;
                }
            }
            _ => {
                // Stepped off ghost house tiles → no longer inside.
                self.in_ghost_house = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};

    const DT: f32 = 1.0 / 60.0;

    /// Open maze with border walls for basic movement tests.
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
        tiles[2][2] = TileType::Wall;
        MazeData { tiles }
    }

    /// Maze with ghost house for house-related tests.
    ///
    /// ```text
    /// row 10: ... E  E  E  E  E ...
    /// row 11: ... E  D  D  D  E ...   (D = GhostDoor)
    /// row 12: ... E  H  H  H  E ...   (H = GhostHouse)
    /// row 13: ... E  H  H  H  E ...
    /// row 14: ... E  E  E  E  E ...
    /// ```
    fn ghost_house_maze() -> MazeData {
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
        // Ghost door row
        tiles[11][12] = TileType::GhostDoor;
        tiles[11][13] = TileType::GhostDoor;
        tiles[11][14] = TileType::GhostDoor;
        // Ghost house interior
        tiles[12][12] = TileType::GhostHouse;
        tiles[12][13] = TileType::GhostHouse;
        tiles[12][14] = TileType::GhostHouse;
        tiles[13][12] = TileType::GhostHouse;
        tiles[13][13] = TileType::GhostHouse;
        tiles[13][14] = TileType::GhostHouse;
        MazeData { tiles }
    }

    /// Maze with a horizontal tunnel (open left and right edges on row 1).
    fn tunnel_maze() -> MazeData {
        let mut tiles = [[TileType::Wall; MAZE_WIDTH]; MAZE_HEIGHT];
        for x in 0..MAZE_WIDTH {
            tiles[1][x] = TileType::Empty;
        }
        MazeData { tiles }
    }

    /// Maze forming a dead end: only Left and Right from (5,5), wall above and below.
    fn dead_end_maze() -> MazeData {
        let mut tiles = [[TileType::Wall; MAZE_WIDTH]; MAZE_HEIGHT];
        // Horizontal corridor at row 5
        for x in 1..MAZE_WIDTH - 1 {
            tiles[5][x] = TileType::Empty;
        }
        MazeData { tiles }
    }

    // ── Construction ──────────────────────────────────────────

    #[test]
    fn new_places_at_grid_position() {
        let ghost = Ghost::new(5, 10);
        assert_eq!(ghost.grid_x(), 5);
        assert_eq!(ghost.grid_y(), 10);
    }

    #[test]
    fn new_starts_stationary() {
        let ghost = Ghost::new(1, 1);
        assert_eq!(ghost.current_dir(), None);
        assert_eq!(ghost.requested_dir(), None);
    }

    #[test]
    fn new_starts_fully_arrived() {
        let ghost = Ghost::new(1, 1);
        assert!((ghost.move_progress() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn new_starts_in_scatter() {
        let ghost = Ghost::new(1, 1);
        assert_eq!(ghost.mode(), GhostMode::Scatter);
    }

    #[test]
    fn new_not_in_house() {
        let ghost = Ghost::new(1, 1);
        assert!(!ghost.in_ghost_house());
    }

    #[test]
    fn new_home_equals_spawn() {
        let ghost = Ghost::new(13, 14);
        assert_eq!(ghost.home_x(), 13);
        assert_eq!(ghost.home_y(), 14);
    }

    #[test]
    fn new_in_house_sets_flag() {
        let ghost = Ghost::new_in_house(13, 14);
        assert!(ghost.in_ghost_house());
        assert_eq!(ghost.grid_x(), 13);
        assert_eq!(ghost.grid_y(), 14);
    }

    // ── Mode speed ────────────────────────────────────────────

    #[test]
    fn chase_speed() {
        assert!((GhostMode::Chase.speed() - NORMAL_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn scatter_speed() {
        assert!((GhostMode::Scatter.speed() - NORMAL_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn frightened_speed() {
        assert!((GhostMode::Frightened.speed() - FRIGHTENED_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn eaten_speed() {
        assert!((GhostMode::Eaten.speed() - EATEN_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn ghost_speed_matches_mode() {
        let mut ghost = Ghost::new(1, 1);
        assert!((ghost.speed() - NORMAL_SPEED).abs() < f32::EPSILON);
        ghost.set_mode(GhostMode::Frightened);
        assert!((ghost.speed() - FRIGHTENED_SPEED).abs() < f32::EPSILON);
        ghost.set_mode(GhostMode::Eaten);
        assert!((ghost.speed() - EATEN_SPEED).abs() < f32::EPSILON);
    }

    // ── Mode transitions ──────────────────────────────────────

    #[test]
    fn set_mode_changes_mode() {
        let mut ghost = Ghost::new(1, 1);
        ghost.set_mode(GhostMode::Chase);
        assert_eq!(ghost.mode(), GhostMode::Chase);
    }

    #[test]
    fn set_same_mode_is_noop() {
        let mut ghost = Ghost::new(1, 1);
        assert_eq!(ghost.mode(), GhostMode::Scatter);
        ghost.set_mode(GhostMode::Scatter);
        assert_eq!(ghost.mode(), GhostMode::Scatter);
    }

    #[test]
    fn scatter_to_chase_reverses_direction() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_direction(Direction::Right);
        // Start moving
        ghost.update(DT, &maze);
        ghost.update(DT, &maze);
        assert_eq!(ghost.current_dir(), Some(Direction::Right));

        ghost.set_mode(GhostMode::Chase);
        assert_eq!(ghost.current_dir(), Some(Direction::Left));
    }

    #[test]
    fn chase_to_scatter_reverses_direction() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        ghost.set_direction(Direction::Right);
        ghost.update(DT, &maze);
        ghost.update(DT, &maze);
        assert_eq!(ghost.current_dir(), Some(Direction::Right));

        ghost.set_mode(GhostMode::Scatter);
        assert_eq!(ghost.current_dir(), Some(Direction::Left));
    }

    #[test]
    fn entering_frightened_reverses_direction() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        ghost.set_direction(Direction::Down);
        ghost.update(DT, &maze);
        ghost.update(DT, &maze);

        ghost.set_mode(GhostMode::Frightened);
        assert_eq!(ghost.current_dir(), Some(Direction::Up));
    }

    #[test]
    fn entering_eaten_does_not_reverse() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        ghost.set_direction(Direction::Right);
        ghost.update(DT, &maze);
        ghost.update(DT, &maze);

        ghost.set_mode(GhostMode::Eaten);
        assert_eq!(ghost.current_dir(), Some(Direction::Right));
    }

    #[test]
    fn frightened_to_chase_does_not_reverse() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Frightened);
        ghost.set_direction(Direction::Right);
        ghost.update(DT, &maze);
        ghost.update(DT, &maze);

        ghost.set_mode(GhostMode::Chase);
        assert_eq!(ghost.current_dir(), Some(Direction::Right));
    }

    #[test]
    fn reverse_mid_tile_swaps_positions() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_direction(Direction::Right);
        // Move partway
        ghost.update(DT, &maze);
        ghost.update(DT, &maze);
        assert!(ghost.move_progress() < 1.0);

        ghost.set_mode(GhostMode::Chase); // triggers reverse from Scatter
        // prev and grid should swap
        assert_eq!(ghost.current_dir(), Some(Direction::Left));
    }

    // ── Basic movement ────────────────────────────────────────

    #[test]
    fn moves_right() {
        let maze = test_maze();
        let mut ghost = Ghost::new(1, 1);
        ghost.set_direction(Direction::Right);

        for _ in 0..10 {
            ghost.update(DT, &maze);
        }
        assert!(ghost.grid_x() >= 2);
        assert_eq!(ghost.grid_y(), 1);
    }

    #[test]
    fn moves_down() {
        let maze = test_maze();
        let mut ghost = Ghost::new(1, 1);
        ghost.set_direction(Direction::Down);

        for _ in 0..10 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.grid_x(), 1);
        assert!(ghost.grid_y() >= 2);
    }

    #[test]
    fn moves_left() {
        let maze = test_maze();
        let mut ghost = Ghost::new(3, 1);
        ghost.set_direction(Direction::Left);

        for _ in 0..10 {
            ghost.update(DT, &maze);
        }
        assert!(ghost.grid_x() <= 2);
    }

    #[test]
    fn moves_up() {
        let maze = test_maze();
        let mut ghost = Ghost::new(1, 3);
        ghost.set_direction(Direction::Up);

        for _ in 0..10 {
            ghost.update(DT, &maze);
        }
        assert!(ghost.grid_y() <= 2);
    }

    #[test]
    fn stationary_without_input() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);

        for _ in 0..100 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.grid_x(), 5);
        assert_eq!(ghost.grid_y(), 5);
    }

    #[test]
    fn continues_through_open_corridor() {
        let maze = test_maze();
        let mut ghost = Ghost::new(1, 1);
        ghost.set_direction(Direction::Right);

        for _ in 0..500 {
            ghost.update(DT, &maze);
        }
        assert!(ghost.grid_x() > 2);
    }

    // ── Wall collision ────────────────────────────────────────

    #[test]
    fn blocked_by_wall() {
        let maze = test_maze();
        let mut ghost = Ghost::new(1, 1);
        ghost.set_direction(Direction::Left);

        for _ in 0..100 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.grid_x(), 1);
        assert_eq!(ghost.grid_y(), 1);
    }

    #[test]
    fn blocked_by_interior_wall() {
        let maze = test_maze();
        let mut ghost = Ghost::new(1, 2);
        ghost.set_direction(Direction::Right);

        for _ in 0..100 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.grid_x(), 1);
        assert_eq!(ghost.grid_y(), 2);
    }

    #[test]
    fn stops_when_hitting_wall() {
        let maze = test_maze();
        let mut ghost = Ghost::new(1, 1);
        ghost.set_direction(Direction::Up);

        for _ in 0..100 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.grid_y(), 1);
        assert_eq!(ghost.current_dir(), None);
    }

    // ── No-reverse rule ───────────────────────────────────────

    #[test]
    fn requested_reverse_ignored_at_boundary() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_direction(Direction::Right);

        // Move to next tile
        for _ in 0..20 {
            ghost.update(DT, &maze);
        }
        // Now request reverse
        ghost.set_direction(Direction::Left);

        let x_before = ghost.grid_x();
        for _ in 0..20 {
            ghost.update(DT, &maze);
        }
        // Should have continued right (reverse ignored), moved further right.
        assert!(ghost.grid_x() >= x_before);
    }

    #[test]
    fn reverse_allowed_at_dead_end() {
        let maze = dead_end_maze();
        // Start at (1, 5) moving left — wall at (0, 5)
        let mut ghost = Ghost::new(1, 5);
        ghost.set_direction(Direction::Left);

        // Advance — should hit the wall and be unable to go up/down (walls)
        for _ in 0..100 {
            ghost.update(DT, &maze);
        }
        // Ghost should be at (1, 5), blocked left by wall
        assert_eq!(ghost.grid_x(), 1);

        // Now request right (reverse) — should be accepted as dead end
        ghost.set_direction(Direction::Right);
        for _ in 0..20 {
            ghost.update(DT, &maze);
        }
        assert!(ghost.grid_x() > 1);
    }

    // ── Ghost house passability ───────────────────────────────

    #[test]
    fn normal_ghost_blocked_by_ghost_door() {
        let maze = ghost_house_maze();
        // Ghost outside house at (13, 10), trying to go down through door at (13, 11)
        let mut ghost = Ghost::new(13, 10);
        ghost.set_direction(Direction::Down);

        for _ in 0..100 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.grid_y(), 10);
    }

    #[test]
    fn normal_ghost_blocked_by_ghost_house() {
        let maze = ghost_house_maze();
        let mut ghost = Ghost::new(11, 12);
        ghost.set_direction(Direction::Right);

        for _ in 0..100 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.grid_x(), 11);
    }

    #[test]
    fn in_house_ghost_can_traverse_door() {
        let maze = ghost_house_maze();
        // Ghost inside house at (13, 12), going up through door at (13, 11)
        let mut ghost = Ghost::new_in_house(13, 12);
        ghost.set_direction(Direction::Up);

        for _ in 0..20 {
            ghost.update(DT, &maze);
        }
        // Should pass through ghost door
        assert!(ghost.grid_y() <= 11);
    }

    #[test]
    fn in_house_ghost_can_move_within_house() {
        let maze = ghost_house_maze();
        let mut ghost = Ghost::new_in_house(12, 12);
        ghost.set_direction(Direction::Right);

        for _ in 0..20 {
            ghost.update(DT, &maze);
        }
        assert!(ghost.grid_x() > 12);
    }

    #[test]
    fn eaten_ghost_can_enter_house() {
        let maze = ghost_house_maze();
        // Eaten ghost outside at (13, 10), going down through door
        let mut ghost = Ghost::new(13, 10);
        ghost.set_mode(GhostMode::Chase);
        ghost.set_mode(GhostMode::Eaten);
        ghost.set_direction(Direction::Down);

        for _ in 0..100 {
            ghost.update(DT, &maze);
        }
        // Should have passed through the door and into the house
        assert!(ghost.grid_y() > 10);
    }

    #[test]
    fn ghost_leaves_house_flag_cleared() {
        let maze = ghost_house_maze();
        let mut ghost = Ghost::new_in_house(13, 12);
        assert!(ghost.in_ghost_house());

        ghost.set_direction(Direction::Up);
        // Move up through door (11) to empty tile (10)
        for _ in 0..200 {
            ghost.update(DT, &maze);
        }
        // Should have left the ghost house area
        assert!(!ghost.in_ghost_house());
    }

    // ── Tunnel wrapping ───────────────────────────────────────

    #[test]
    fn wraps_left_to_right() {
        let maze = tunnel_maze();
        let mut ghost = Ghost::new(0, 1);
        ghost.set_direction(Direction::Left);

        for _ in 0..8 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.grid_x(), MAZE_WIDTH - 1);
    }

    #[test]
    fn wraps_right_to_left() {
        let maze = tunnel_maze();
        let mut ghost = Ghost::new(MAZE_WIDTH - 1, 1);
        ghost.set_direction(Direction::Right);

        for _ in 0..8 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.grid_x(), 0);
    }

    // ── World position ────────────────────────────────────────

    #[test]
    fn world_position_at_grid_when_stationary() {
        let ghost = Ghost::new(5, 10);
        let (wx, wz) = ghost.world_position(0.0);
        assert!((wx - 5.0).abs() < 1e-5);
        assert!((wz - 10.0).abs() < 1e-5);
    }

    #[test]
    fn world_position_interpolates_during_move() {
        let maze = test_maze();
        let mut ghost = Ghost::new(1, 1);
        ghost.set_direction(Direction::Right);

        ghost.update(DT, &maze);
        ghost.update(DT, &maze);
        let (wx, _) = ghost.world_position(0.0);
        assert!(wx > 1.0 && wx < 2.0, "wx = {}", wx);
    }

    #[test]
    fn world_position_at_full_progress_equals_grid() {
        let ghost = Ghost::new(3, 7);
        let (wx, wz) = ghost.world_position(0.5);
        assert!((wx - 3.0).abs() < 1e-5);
        assert!((wz - 7.0).abs() < 1e-5);
    }

    // ── Collider ──────────────────────────────────────────────

    #[test]
    fn collider_position_matches() {
        let ghost = Ghost::new(5, 10);
        let c = ghost.to_collider();
        assert_eq!(c.grid_x, 5);
        assert_eq!(c.grid_y, 10);
    }

    #[test]
    fn collider_not_frightened_by_default() {
        let ghost = Ghost::new(5, 10);
        assert!(!ghost.to_collider().frightened);
    }

    #[test]
    fn collider_frightened_when_mode_is_frightened() {
        let mut ghost = Ghost::new(5, 10);
        ghost.set_mode(GhostMode::Frightened);
        assert!(ghost.to_collider().frightened);
    }

    #[test]
    fn collider_not_frightened_in_eaten_mode() {
        let mut ghost = Ghost::new(5, 10);
        ghost.set_mode(GhostMode::Frightened);
        ghost.set_mode(GhostMode::Eaten);
        assert!(!ghost.to_collider().frightened);
    }

    // ── Reached home ──────────────────────────────────────────

    #[test]
    fn reached_home_when_eaten_at_home_tile() {
        let ghost = Ghost {
            grid_x: 13,
            grid_y: 14,
            prev_x: 13,
            prev_y: 13,
            current_dir: Some(Direction::Down),
            requested_dir: None,
            move_progress: 1.0,
            mode: GhostMode::Eaten,
            home_x: 13,
            home_y: 14,
            in_ghost_house: false,
        };
        assert!(ghost.reached_home());
    }

    #[test]
    fn not_reached_home_when_not_eaten() {
        let ghost = Ghost::new(5, 5);
        assert!(!ghost.reached_home());
    }

    #[test]
    fn not_reached_home_when_not_at_home_tile() {
        let mut ghost = Ghost::new(13, 14);
        ghost.set_mode(GhostMode::Chase);
        ghost.set_mode(GhostMode::Eaten);
        // Move away from home
        let maze = test_maze();
        ghost.set_direction(Direction::Right);
        for _ in 0..20 {
            ghost.update(DT, &maze);
        }
        assert!(!ghost.reached_home());
    }

    #[test]
    fn not_reached_home_mid_move() {
        let ghost = Ghost {
            grid_x: 13,
            grid_y: 14,
            prev_x: 13,
            prev_y: 13,
            current_dir: Some(Direction::Down),
            requested_dir: None,
            move_progress: 0.5,
            mode: GhostMode::Eaten,
            home_x: 13,
            home_y: 14,
            in_ghost_house: false,
        };
        assert!(!ghost.reached_home());
    }

    // ── Direction request ─────────────────────────────────────

    #[test]
    fn set_direction_stores_request() {
        let mut ghost = Ghost::new(5, 5);
        ghost.set_direction(Direction::Up);
        assert_eq!(ghost.requested_dir(), Some(Direction::Up));
    }

    #[test]
    fn requested_dir_applied_at_tile_boundary() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_direction(Direction::Right);

        // Move to reach a boundary
        for _ in 0..20 {
            ghost.update(DT, &maze);
        }
        // Request turn
        ghost.set_direction(Direction::Down);
        for _ in 0..200 {
            ghost.update(DT, &maze);
        }
        // Should have turned down at some point
        assert!(ghost.grid_y() > 5);
    }

    #[test]
    fn requested_dir_cleared_after_application() {
        let maze = test_maze();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_direction(Direction::Right);

        // Run enough to apply the direction
        for _ in 0..20 {
            ghost.update(DT, &maze);
        }
        assert_eq!(ghost.requested_dir(), None);
    }

    // ── Classic maze integration ──────────────────────────────

    #[test]
    fn ghost_in_classic_maze_ghost_house() {
        let data = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/maze/classic.json"),
        )
        .expect("classic.json should exist");
        let maze = MazeData::from_json(&data).expect("should parse");

        // Find a ghost house tile
        let mut house_pos = None;
        for y in 0..MAZE_HEIGHT {
            for x in 0..MAZE_WIDTH {
                if maze.tiles[y][x] == TileType::GhostHouse {
                    house_pos = Some((x, y));
                    break;
                }
            }
            if house_pos.is_some() {
                break;
            }
        }
        let (hx, hy) = house_pos.expect("classic maze should have ghost house");

        let ghost = Ghost::new_in_house(hx, hy);
        assert!(ghost.in_ghost_house());
        assert_eq!(ghost.mode(), GhostMode::Scatter);
    }
}
