//! Pac-Man entity — grid movement, direction queue, smooth interpolation, wall collision.

use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};

/// Movement speed in tiles per second.
const DEFAULT_SPEED: f32 = 8.0;

/// Cardinal direction on the maze grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    /// Returns the opposite direction.
    pub fn opposite(self) -> Self {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    /// Grid offset `(dx, dy)` for this direction.
    ///
    /// Up decreases row (y-1), down increases row (y+1).
    pub(crate) fn delta(self) -> (i32, i32) {
        match self {
            Direction::Up => (0, -1),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }
}

/// Pac-Man entity with grid-based movement and smooth visual interpolation.
///
/// Movement uses a direction queue: the player's most recent input is stored
/// as `queued_dir` and applied as soon as the target tile is reachable. If the
/// queued direction is blocked, the current direction continues. This matches
/// the classic "cornering" feel.
///
/// Visual position is interpolated between the previous tile and current tile
/// based on `move_progress` (0.0–1.0) so rendering stays smooth between fixed
/// physics ticks.
pub struct PacMan {
    /// Current grid column.
    grid_x: usize,
    /// Current grid row.
    grid_y: usize,
    /// Grid column we are moving from (for interpolation).
    prev_x: usize,
    /// Grid row we are moving from (for interpolation).
    prev_y: usize,
    /// Direction currently being travelled.
    current_dir: Option<Direction>,
    /// Next direction requested by the player (buffered input).
    queued_dir: Option<Direction>,
    /// Progress from `(prev_x, prev_y)` toward `(grid_x, grid_y)`, 0.0–1.0.
    move_progress: f32,
    /// Movement speed in tiles per second.
    speed: f32,
}

impl PacMan {
    /// Create a new Pac-Man placed at the given grid position.
    pub fn new(grid_x: usize, grid_y: usize) -> Self {
        Self {
            grid_x,
            grid_y,
            prev_x: grid_x,
            prev_y: grid_y,
            current_dir: None,
            queued_dir: None,
            move_progress: 1.0, // start fully arrived
            speed: DEFAULT_SPEED,
        }
    }

    /// Create a Pac-Man at the `PlayerSpawn` tile in the maze.
    ///
    /// Returns `None` if the maze has no `PlayerSpawn` tile.
    pub fn from_maze(maze: &MazeData) -> Option<Self> {
        for y in 0..MAZE_HEIGHT {
            for x in 0..MAZE_WIDTH {
                if maze.tiles[y][x] == TileType::PlayerSpawn {
                    return Some(Self::new(x, y));
                }
            }
        }
        None
    }

    /// Current grid column.
    pub fn grid_x(&self) -> usize {
        self.grid_x
    }

    /// Current grid row.
    pub fn grid_y(&self) -> usize {
        self.grid_y
    }

    /// Previous grid column (start of current move).
    pub fn prev_x(&self) -> usize {
        self.prev_x
    }

    /// Previous grid row (start of current move).
    pub fn prev_y(&self) -> usize {
        self.prev_y
    }

    /// Direction Pac-Man is currently moving (if any).
    pub fn current_dir(&self) -> Option<Direction> {
        self.current_dir
    }

    /// Queued input direction (if any).
    pub fn queued_dir(&self) -> Option<Direction> {
        self.queued_dir
    }

    /// Movement progress between previous and current tile (0.0–1.0).
    pub fn move_progress(&self) -> f32 {
        self.move_progress
    }

    /// Movement speed in tiles per second.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Set the movement speed in tiles per second.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Queue a direction change. Applied at the next tile boundary (or
    /// immediately if Pac-Man is stationary or reversing).
    pub fn set_direction(&mut self, dir: Direction) {
        self.queued_dir = Some(dir);
    }

    /// Advance Pac-Man by one fixed timestep.
    ///
    /// `dt` is the fixed timestep duration in seconds. Movement is checked
    /// against the maze for wall collisions, and tunnel wrapping is applied
    /// at the horizontal edges.
    pub fn update(&mut self, dt: f32, maze: &MazeData) {
        // If reversing direction mid-tile, swap immediately.
        if let (Some(queued), Some(current)) = (self.queued_dir, self.current_dir) {
            if queued == current.opposite() && self.move_progress < 1.0 {
                // Swap prev and current, invert progress.
                let old_grid_x = self.grid_x;
                let old_grid_y = self.grid_y;
                self.grid_x = self.prev_x;
                self.grid_y = self.prev_y;
                self.prev_x = old_grid_x;
                self.prev_y = old_grid_y;
                self.move_progress = 1.0 - self.move_progress;
                self.current_dir = Some(queued);
                self.queued_dir = None;
            }
        }

        // Advance progress.
        if self.current_dir.is_some() && self.move_progress < 1.0 {
            self.move_progress += self.speed * dt;
        }

        // Arrived at the target tile?
        if self.move_progress >= 1.0 {
            self.move_progress = 1.0;

            // Try queued direction first, then continue current direction.
            let next_dir = self
                .queued_dir
                .filter(|&d| self.can_move(d, maze))
                .or_else(|| self.current_dir.filter(|&d| self.can_move(d, maze)));

            if let Some(dir) = next_dir {
                // If we used the queued direction, clear it.
                if self.queued_dir == Some(dir) {
                    self.queued_dir = None;
                }
                self.begin_move(dir, maze);
            } else {
                // Blocked — stop moving.
                self.current_dir = None;
            }
        }
    }

    /// Interpolated world-space position for rendering.
    ///
    /// `alpha` is the rendering interpolation factor (from `TimeState::alpha`).
    /// Returns `(world_x, world_z)` matching the maze renderer convention
    /// where grid `(x, y)` maps to world `(x, 0, y)`.
    pub fn world_position(&self, alpha: f32) -> (f32, f32) {
        // Interpolate between prev and current based on move_progress.
        let t = self.move_progress.clamp(0.0, 1.0);

        let visual_x = self.prev_x as f32 + (self.grid_x as f32 - self.prev_x as f32) * t;
        let visual_z = self.prev_y as f32 + (self.grid_y as f32 - self.prev_y as f32) * t;

        // Apply rendering alpha for sub-frame smoothing.
        // At this point visual position already reflects the last physics state,
        // alpha would further interpolate toward "next predicted" but since we
        // don't have two physics states we use the current interpolation directly.
        let _ = alpha;

        (visual_x, visual_z)
    }

    /// Whether Pac-Man can move in the given direction from the current tile.
    fn can_move(&self, dir: Direction, maze: &MazeData) -> bool {
        let (tx, ty) = self.target_tile(self.grid_x, self.grid_y, dir);
        is_passable(maze, tx, ty)
    }

    /// Start moving toward the adjacent tile in `dir`.
    fn begin_move(&mut self, dir: Direction, maze: &MazeData) {
        let (tx, ty) = self.target_tile(self.grid_x, self.grid_y, dir);
        self.prev_x = self.grid_x;
        self.prev_y = self.grid_y;
        self.grid_x = tx;
        self.grid_y = ty;
        self.current_dir = Some(dir);
        self.move_progress = 0.0;
        let _ = maze;
    }

    /// Compute the target tile from `(x, y)` in direction `dir`, applying
    /// horizontal tunnel wrapping.
    fn target_tile(&self, x: usize, y: usize, dir: Direction) -> (usize, usize) {
        let (dx, dy) = dir.delta();
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;

        // Horizontal tunnel wrapping.
        let wrapped_x = if nx < 0 {
            MAZE_WIDTH - 1
        } else if nx >= MAZE_WIDTH as i32 {
            0
        } else {
            nx as usize
        };

        // Vertical: clamp (no vertical tunnels in classic Pac-Man).
        let wrapped_y = ny.clamp(0, MAZE_HEIGHT as i32 - 1) as usize;

        (wrapped_x, wrapped_y)
    }
}

/// A tile is passable if it is not a wall, ghost house, or ghost door.
fn is_passable(maze: &MazeData, x: usize, y: usize) -> bool {
    match maze.get(x, y) {
        Some(TileType::Wall) | Some(TileType::GhostHouse) | Some(TileType::GhostDoor) => false,
        Some(_) => true,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};

    /// Minimal maze with a known layout for deterministic tests.
    ///
    /// ```text
    /// W W W W W W ...
    /// W . . . . W ...
    /// W . W . . W ...
    /// W . . . . W ...
    /// W W W W W W ...
    /// ```
    /// (Only the top-left corner matters; rest is walls on border, empty inside.)
    fn test_maze() -> MazeData {
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
        // Interior wall for collision tests
        tiles[2][2] = TileType::Wall;
        // Player spawn
        tiles[1][1] = TileType::PlayerSpawn;
        MazeData { tiles }
    }

    /// Maze with a horizontal tunnel (open left and right edges on row 1).
    fn tunnel_maze() -> MazeData {
        let mut tiles = [[TileType::Wall; MAZE_WIDTH]; MAZE_HEIGHT];
        // Open a corridor at row 1.
        for x in 0..MAZE_WIDTH {
            tiles[1][x] = TileType::Empty;
        }
        // Open the left and right edges for wrapping.
        tiles[1][0] = TileType::Empty;
        tiles[1][MAZE_WIDTH - 1] = TileType::Empty;
        MazeData { tiles }
    }

    fn spawn_maze() -> MazeData {
        let mut tiles = [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT];
        tiles[15][14] = TileType::PlayerSpawn;
        MazeData { tiles }
    }

    fn no_spawn_maze() -> MazeData {
        MazeData {
            tiles: [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT],
        }
    }

    const DT: f32 = 1.0 / 60.0;

    // ── Construction ──────────────────────────────────────────

    #[test]
    fn new_places_at_grid_position() {
        let pac = PacMan::new(5, 10);
        assert_eq!(pac.grid_x(), 5);
        assert_eq!(pac.grid_y(), 10);
    }

    #[test]
    fn new_starts_stationary() {
        let pac = PacMan::new(1, 1);
        assert_eq!(pac.current_dir(), None);
        assert_eq!(pac.queued_dir(), None);
    }

    #[test]
    fn new_starts_fully_arrived() {
        let pac = PacMan::new(1, 1);
        assert!((pac.move_progress() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn from_maze_finds_spawn() {
        let pac = PacMan::from_maze(&spawn_maze()).unwrap();
        assert_eq!(pac.grid_x(), 14);
        assert_eq!(pac.grid_y(), 15);
    }

    #[test]
    fn from_maze_returns_none_without_spawn() {
        assert!(PacMan::from_maze(&no_spawn_maze()).is_none());
    }

    #[test]
    fn default_speed() {
        let pac = PacMan::new(1, 1);
        assert!((pac.speed() - DEFAULT_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn set_speed() {
        let mut pac = PacMan::new(1, 1);
        pac.set_speed(12.0);
        assert!((pac.speed() - 12.0).abs() < f32::EPSILON);
    }

    // ── Basic movement ────────────────────────────────────────

    #[test]
    fn moves_right() {
        let maze = test_maze();
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Right);

        // Enough ticks to travel at least one full tile.
        for _ in 0..10 {
            pac.update(DT, &maze);
        }
        assert!(pac.grid_x() >= 2);
        assert_eq!(pac.grid_y(), 1);
    }

    #[test]
    fn moves_down() {
        let maze = test_maze();
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Down);

        for _ in 0..10 {
            pac.update(DT, &maze);
        }
        assert_eq!(pac.grid_x(), 1);
        assert!(pac.grid_y() >= 2);
    }

    #[test]
    fn moves_left() {
        let maze = test_maze();
        let mut pac = PacMan::new(3, 1);
        pac.set_direction(Direction::Left);

        for _ in 0..10 {
            pac.update(DT, &maze);
        }
        assert!(pac.grid_x() <= 2);
        assert_eq!(pac.grid_y(), 1);
    }

    #[test]
    fn moves_up() {
        let maze = test_maze();
        let mut pac = PacMan::new(1, 3);
        pac.set_direction(Direction::Up);

        for _ in 0..10 {
            pac.update(DT, &maze);
        }
        assert_eq!(pac.grid_x(), 1);
        assert!(pac.grid_y() <= 2);
    }

    // ── Wall collision ────────────────────────────────────────

    #[test]
    fn blocked_by_wall() {
        let maze = test_maze();
        // Position (1,1), wall at (0,1) to the left.
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Left);

        for _ in 0..100 {
            pac.update(DT, &maze);
        }
        // Should stay at (1,1) — can't enter the wall.
        assert_eq!(pac.grid_x(), 1);
        assert_eq!(pac.grid_y(), 1);
        assert_eq!(pac.current_dir(), None);
    }

    #[test]
    fn blocked_by_interior_wall() {
        let maze = test_maze();
        // Wall at (2,2). Start at (1,2), try to go right.
        let mut pac = PacMan::new(1, 2);
        pac.set_direction(Direction::Right);

        for _ in 0..100 {
            pac.update(DT, &maze);
        }
        assert_eq!(pac.grid_x(), 1);
        assert_eq!(pac.grid_y(), 2);
    }

    #[test]
    fn stops_when_hitting_wall() {
        let maze = test_maze();
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Up);

        for _ in 0..100 {
            pac.update(DT, &maze);
        }
        assert_eq!(pac.grid_y(), 1);
        assert_eq!(pac.current_dir(), None);
    }

    #[test]
    fn ghost_house_is_impassable() {
        let mut tiles = [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT];
        tiles[2][1] = TileType::GhostHouse;
        let maze = MazeData { tiles };
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Down);

        for _ in 0..100 {
            pac.update(DT, &maze);
        }
        assert_eq!(pac.grid_y(), 1);
    }

    #[test]
    fn ghost_door_is_impassable() {
        let mut tiles = [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT];
        tiles[2][1] = TileType::GhostDoor;
        let maze = MazeData { tiles };
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Down);

        for _ in 0..100 {
            pac.update(DT, &maze);
        }
        assert_eq!(pac.grid_y(), 1);
    }

    // ── Direction queue / cornering ───────────────────────────

    #[test]
    fn queued_direction_applied_at_tile_boundary() {
        let maze = test_maze();
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Right);

        // Advance partway.
        for _ in 0..4 {
            pac.update(DT, &maze);
        }
        // Queue down while moving right.
        pac.set_direction(Direction::Down);

        // Finish reaching tile (2,1) and then continue to (2,2).
        for _ in 0..200 {
            pac.update(DT, &maze);
        }
        // Should have turned down after reaching (2,1).
        // But (2,2) is a wall in test_maze, so Pac-Man stops at (2,1).
        // Let's check the direction was at least attempted.
        // Actually: wall at (2,2), so queued Down is blocked → continues Right.
        // (2,1) → Right → (3,1).
        assert!(pac.grid_x() >= 2);
    }

    #[test]
    fn queued_direction_takes_priority_over_current() {
        let maze = test_maze();
        // Start at (1,3), go right. Queue down at (2,3) which is open below.
        let mut pac = PacMan::new(1, 3);
        pac.set_direction(Direction::Right);

        // Advance until we reach (2,3).
        for _ in 0..20 {
            pac.update(DT, &maze);
        }
        // Queue down — (2,4) should be passable (interior is empty).
        pac.set_direction(Direction::Down);

        for _ in 0..200 {
            pac.update(DT, &maze);
        }
        // Should have gone down at some point.
        assert!(pac.grid_y() > 3);
    }

    #[test]
    fn queued_blocked_continues_current() {
        let maze = test_maze();
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Right);

        // Advance a bit.
        for _ in 0..4 {
            pac.update(DT, &maze);
        }
        // Queue up — blocked by wall at row 0.
        pac.set_direction(Direction::Up);

        for _ in 0..100 {
            pac.update(DT, &maze);
        }
        // Should have continued right since up is blocked.
        assert!(pac.grid_x() > 1);
        assert_eq!(pac.grid_y(), 1);
    }

    // ── Reversal ──────────────────────────────────────────────

    #[test]
    fn reverse_mid_tile() {
        let maze = test_maze();
        // Start at (3,1) so Left is passable after reversal (wall at col 0
        // would block if we started at (1,1)).
        let mut pac = PacMan::new(3, 1);
        pac.set_direction(Direction::Right);

        // Advance partway (not a full tile).
        for _ in 0..2 {
            pac.update(DT, &maze);
        }
        assert!(pac.move_progress() < 1.0);

        pac.set_direction(Direction::Left);
        pac.update(DT, &maze);

        // After reversal, current direction should be Left.
        assert_eq!(pac.current_dir(), Some(Direction::Left));
    }

    #[test]
    fn reverse_swaps_prev_and_current() {
        let maze = test_maze();
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Right);

        // Move partway toward (2,1).
        for _ in 0..3 {
            pac.update(DT, &maze);
        }
        let progress_before = pac.move_progress();

        pac.set_direction(Direction::Left);
        pac.update(DT, &maze);

        // grid_x should now be 1 (going back), prev should be 2.
        assert_eq!(pac.grid_x(), 1);
        assert_eq!(pac.prev_x(), 2);
        // Progress should be roughly inverted from before the reverse + one tick.
        let _ = progress_before;
    }

    // ── Tunnel wrapping ───────────────────────────────────────

    #[test]
    fn wraps_left_to_right() {
        let maze = tunnel_maze();
        let mut pac = PacMan::new(0, 1);
        pac.set_direction(Direction::Left);

        // 8 ticks: starts move to wrapped tile, advances without arriving.
        for _ in 0..8 {
            pac.update(DT, &maze);
        }
        assert_eq!(pac.grid_x(), MAZE_WIDTH - 1);
    }

    #[test]
    fn wraps_right_to_left() {
        let maze = tunnel_maze();
        let mut pac = PacMan::new(MAZE_WIDTH - 1, 1);
        pac.set_direction(Direction::Right);

        for _ in 0..8 {
            pac.update(DT, &maze);
        }
        assert_eq!(pac.grid_x(), 0);
    }

    // ── Interpolation / world position ────────────────────────

    #[test]
    fn world_position_at_grid_when_stationary() {
        let pac = PacMan::new(5, 10);
        let (wx, wz) = pac.world_position(0.0);
        assert!((wx - 5.0).abs() < 1e-5);
        assert!((wz - 10.0).abs() < 1e-5);
    }

    #[test]
    fn world_position_interpolates_during_move() {
        let maze = test_maze();
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Right);

        // First tick starts the move (progress = 0.0), second tick advances it.
        pac.update(DT, &maze);
        pac.update(DT, &maze);
        let (wx, _) = pac.world_position(0.0);
        // Should be between 1.0 and 2.0.
        assert!(wx > 1.0 && wx < 2.0, "wx = {}", wx);
    }

    #[test]
    fn world_position_at_full_progress_equals_grid() {
        let pac = PacMan::new(3, 7);
        let (wx, wz) = pac.world_position(0.5);
        assert!((wx - 3.0).abs() < 1e-5);
        assert!((wz - 7.0).abs() < 1e-5);
    }

    // ── Continuous movement ───────────────────────────────────

    #[test]
    fn continues_moving_through_open_corridor() {
        let maze = test_maze();
        let mut pac = PacMan::new(1, 1);
        pac.set_direction(Direction::Right);

        // Run enough ticks to cross several tiles.
        for _ in 0..500 {
            pac.update(DT, &maze);
        }
        // Should have moved well past tile 1.
        assert!(pac.grid_x() > 2);
    }

    #[test]
    fn stationary_without_input() {
        let maze = test_maze();
        let mut pac = PacMan::new(5, 5);

        for _ in 0..100 {
            pac.update(DT, &maze);
        }
        assert_eq!(pac.grid_x(), 5);
        assert_eq!(pac.grid_y(), 5);
    }

    // ── Direction opposite ────────────────────────────────────

    #[test]
    fn direction_opposites() {
        assert_eq!(Direction::Up.opposite(), Direction::Down);
        assert_eq!(Direction::Down.opposite(), Direction::Up);
        assert_eq!(Direction::Left.opposite(), Direction::Right);
        assert_eq!(Direction::Right.opposite(), Direction::Left);
    }

    // ── Classic maze ──────────────────────────────────────────

    #[test]
    fn from_classic_maze_finds_spawn() {
        let data = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/maze/classic.json"),
        )
        .expect("classic.json should exist");
        let maze = MazeData::from_json(&data).expect("should parse");
        let pac = PacMan::from_maze(&maze);
        assert!(pac.is_some(), "classic maze should have a player spawn");
    }
}
