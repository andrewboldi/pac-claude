//! Ghost base entity — state machine with Chase/Scatter/Frightened/Eaten states, grid movement.

use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};
use crate::pacman::Direction;

/// Normal ghost movement speed in tiles per second.
const NORMAL_SPEED: f32 = 7.0;
/// Speed when frightened.
const FRIGHTENED_SPEED: f32 = 4.0;
/// Speed when eaten (fast return to ghost house).
const EATEN_SPEED: f32 = 14.0;

/// Ghost behavior state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostState {
    /// Actively targeting Pac-Man (target depends on ghost personality).
    Chase,
    /// Retreating to home corner.
    Scatter,
    /// Vulnerable after Pac-Man eats a power pellet.
    Frightened,
    /// Eyes only — returning to the ghost house after being eaten.
    Eaten,
}

/// Alias used by AI modules and the mode controller.
pub type GhostMode = GhostState;

/// Identifies which ghost this is (determines AI personality in downstream modules).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostId {
    Blinky,
    Pinky,
    Inky,
    Clyde,
}

/// A ghost entity with grid-based movement and a four-state behavior machine.
///
/// Movement mirrors [`crate::pacman::PacMan`] — grid coordinates with smooth
/// interpolation via `move_progress` — but differs in passability rules:
/// ghosts can traverse `GhostHouse` tiles, and can pass through `GhostDoor`
/// when in `Eaten` state or when still inside the house.
///
/// The AI targeting logic (which direction to choose at intersections) is
/// **not** part of this base entity — it lives in the per-ghost AI modules
/// (Blinky, Pinky, Inky, Clyde). External code calls [`Ghost::set_direction`]
/// to steer the ghost.
pub struct Ghost {
    id: GhostId,
    state: GhostState,
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
    /// Next direction chosen by AI (applied at tile boundary).
    queued_dir: Option<Direction>,
    /// Progress from `(prev_x, prev_y)` toward `(grid_x, grid_y)`, 0.0–1.0.
    move_progress: f32,
    /// Movement speed in Chase/Scatter (tiles per second).
    normal_speed: f32,
    /// Movement speed when Frightened.
    frightened_speed: f32,
    /// Movement speed when Eaten (returning to house).
    eaten_speed: f32,
    /// Ghost house spawn position (return point when eaten).
    spawn_x: usize,
    /// Ghost house spawn row.
    spawn_y: usize,
    /// Whether the ghost is currently inside the ghost house area.
    in_house: bool,
}

impl Ghost {
    /// Create a ghost at the given grid position, outside the ghost house.
    ///
    /// Initial state is `Scatter` (classic Pac-Man starts with a scatter phase).
    pub fn new(id: GhostId, grid_x: usize, grid_y: usize) -> Self {
        Self {
            id,
            state: GhostState::Scatter,
            grid_x,
            grid_y,
            prev_x: grid_x,
            prev_y: grid_y,
            current_dir: None,
            queued_dir: None,
            move_progress: 1.0, // start fully arrived
            normal_speed: NORMAL_SPEED,
            frightened_speed: FRIGHTENED_SPEED,
            eaten_speed: EATEN_SPEED,
            spawn_x: grid_x,
            spawn_y: grid_y,
            in_house: false,
        }
    }

    /// Create a ghost spawned inside the ghost house.
    ///
    /// Identical to [`Ghost::new`] but sets `in_house` to `true`, allowing the
    /// ghost to exit through the ghost door.
    pub fn new_in_house(id: GhostId, grid_x: usize, grid_y: usize) -> Self {
        let mut ghost = Self::new(id, grid_x, grid_y);
        ghost.in_house = true;
        ghost
    }

    // ── Getters ──────────────────────────────────────────────

    pub fn id(&self) -> GhostId {
        self.id
    }

    pub fn state(&self) -> GhostState {
        self.state
    }

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

    pub fn queued_dir(&self) -> Option<Direction> {
        self.queued_dir
    }

    pub fn move_progress(&self) -> f32 {
        self.move_progress
    }

    pub fn in_house(&self) -> bool {
        self.in_house
    }

    pub fn spawn_x(&self) -> usize {
        self.spawn_x
    }

    pub fn spawn_y(&self) -> usize {
        self.spawn_y
    }

    // ── Aliases for AI modules ──────────────────────────────

    /// Alias for [`state`](Ghost::state) used by AI modules.
    pub fn mode(&self) -> GhostState {
        self.state
    }

    /// Alias for [`spawn_x`](Ghost::spawn_x).
    pub fn home_x(&self) -> usize {
        self.spawn_x
    }

    /// Alias for [`spawn_y`](Ghost::spawn_y).
    pub fn home_y(&self) -> usize {
        self.spawn_y
    }

    /// Alias for [`in_house`](Ghost::in_house).
    pub fn in_ghost_house(&self) -> bool {
        self.in_house
    }

    /// Alias for [`queued_dir`](Ghost::queued_dir).
    pub fn requested_dir(&self) -> Option<Direction> {
        self.queued_dir
    }

    /// Whether the ghost can move in `dir` from the current tile.
    pub fn is_direction_passable(&self, dir: Direction, maze: &MazeData) -> bool {
        let (tx, ty) = self.target_coords(self.grid_x, self.grid_y, dir);
        ghost_is_passable(maze, tx, ty, self.state, self.in_house)
    }

    /// Grid coordinates of the adjacent tile in `dir` (with tunnel wrapping).
    pub fn neighbor_tile(&self, dir: Direction) -> (usize, usize) {
        self.target_coords(self.grid_x, self.grid_y, dir)
    }

    /// Direct state setter used by the mode controller.
    pub fn set_mode(&mut self, mode: GhostState) {
        self.state = mode;
    }

    /// Override normal movement speed (used by Blinky Cruise Elroy).
    pub fn set_speed_override(&mut self, speed: f32) {
        self.normal_speed = speed;
    }

    /// Restore normal movement speed to the default.
    pub fn clear_speed_override(&mut self) {
        self.normal_speed = NORMAL_SPEED;
    }

    /// Current effective speed in tiles per second, based on state.
    pub fn speed(&self) -> f32 {
        match self.state {
            GhostState::Frightened => self.frightened_speed,
            GhostState::Eaten => self.eaten_speed,
            GhostState::Chase | GhostState::Scatter => self.normal_speed,
        }
    }

    // ── Setters ──────────────────────────────────────────────

    /// Override the three speed values (normal, frightened, eaten).
    pub fn set_speeds(&mut self, normal: f32, frightened: f32, eaten: f32) {
        self.normal_speed = normal;
        self.frightened_speed = frightened;
        self.eaten_speed = eaten;
    }

    /// Queue a direction for the ghost to take at the next tile boundary.
    pub fn set_direction(&mut self, dir: Direction) {
        self.queued_dir = Some(dir);
    }

    // ── State transitions ────────────────────────────────────

    /// Transition to Chase state.
    ///
    /// Reverses direction if coming from Scatter (classic Pac-Man rule).
    /// No reversal when exiting Frightened — that reversal already happened
    /// on entry.
    pub fn enter_chase(&mut self) {
        let prev = self.state;
        self.state = GhostState::Chase;
        if prev == GhostState::Scatter {
            self.reverse();
        }
    }

    /// Transition to Scatter state. Reverses direction from Chase.
    pub fn enter_scatter(&mut self) {
        let prev = self.state;
        self.state = GhostState::Scatter;
        if prev == GhostState::Chase {
            self.reverse();
        }
    }

    /// Transition to Frightened state. Reverses direction.
    ///
    /// No-op if the ghost is already Eaten (eyes can't be frightened).
    pub fn enter_frightened(&mut self) {
        if self.state == GhostState::Eaten {
            return;
        }
        self.state = GhostState::Frightened;
        self.reverse();
    }

    /// Transition to Eaten state (Pac-Man ate this ghost).
    ///
    /// No direction reversal — the ghost immediately targets the ghost house.
    pub fn enter_eaten(&mut self) {
        self.state = GhostState::Eaten;
    }

    /// Reset the ghost to its spawn position inside the ghost house.
    ///
    /// Called when the eaten ghost reaches the ghost house. Resets position,
    /// direction, and state to Scatter.
    pub fn respawn(&mut self) {
        self.grid_x = self.spawn_x;
        self.grid_y = self.spawn_y;
        self.prev_x = self.spawn_x;
        self.prev_y = self.spawn_y;
        self.current_dir = None;
        self.queued_dir = None;
        self.move_progress = 1.0;
        self.in_house = true;
        self.state = GhostState::Scatter;
    }

    // ── Movement ─────────────────────────────────────────────

    /// Advance the ghost by one fixed timestep.
    ///
    /// `dt` is the fixed timestep duration in seconds. Movement is checked
    /// against the maze for passability, and tunnel wrapping is applied at
    /// the horizontal edges.
    pub fn update(&mut self, dt: f32, maze: &MazeData) {
        // Handle mid-tile reversal (queued direction is opposite of current).
        if let (Some(queued), Some(current)) = (self.queued_dir, self.current_dir) {
            if queued == current.opposite() && self.move_progress < 1.0 {
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

        // Advance progress toward target tile.
        let speed = self.speed();
        if self.current_dir.is_some() && self.move_progress < 1.0 {
            self.move_progress += speed * dt;
        }

        // Arrived at the target tile?
        if self.move_progress >= 1.0 {
            self.move_progress = 1.0;

            // Update in_house flag based on the tile we just arrived at.
            match maze.get(self.grid_x, self.grid_y) {
                Some(TileType::GhostHouse) => self.in_house = true,
                Some(TileType::GhostDoor) => {} // transitioning — don't change flag
                _ => self.in_house = false,
            }

            // Try queued direction first, then continue current.
            let next_dir = self
                .queued_dir
                .filter(|&d| self.can_move(d, maze))
                .or_else(|| self.current_dir.filter(|&d| self.can_move(d, maze)));

            if let Some(dir) = next_dir {
                if self.queued_dir == Some(dir) {
                    self.queued_dir = None;
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
    pub fn world_position(&self, alpha: f32) -> (f32, f32) {
        let t = self.move_progress.clamp(0.0, 1.0);
        let visual_x = self.prev_x as f32 + (self.grid_x as f32 - self.prev_x as f32) * t;
        let visual_z = self.prev_y as f32 + (self.grid_y as f32 - self.prev_y as f32) * t;
        let _ = alpha;
        (visual_x, visual_z)
    }

    // ── Internal helpers ─────────────────────────────────────

    /// Whether the ghost can move in the given direction from the current tile.
    fn can_move(&self, dir: Direction, maze: &MazeData) -> bool {
        let (tx, ty) = self.target_coords(self.grid_x, self.grid_y, dir);
        ghost_is_passable(maze, tx, ty, self.state, self.in_house)
    }

    /// Start moving toward the adjacent tile in `dir`.
    fn begin_move(&mut self, dir: Direction, maze: &MazeData) {
        let (tx, ty) = self.target_coords(self.grid_x, self.grid_y, dir);
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
    fn target_coords(&self, x: usize, y: usize, dir: Direction) -> (usize, usize) {
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

    /// Reverse the ghost's current direction mid-tile.
    fn reverse(&mut self) {
        if let Some(dir) = self.current_dir {
            if self.move_progress < 1.0 {
                // Mid-tile: swap prev/current positions and invert progress.
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
        self.queued_dir = None;
    }
}

/// Ghost-specific passability check.
///
/// Unlike Pac-Man, ghosts can walk on `GhostHouse` tiles, and can pass through
/// `GhostDoor` when in `Eaten` state (returning home) or when inside the house
/// (exiting).
fn ghost_is_passable(
    maze: &MazeData,
    x: usize,
    y: usize,
    state: GhostState,
    in_house: bool,
) -> bool {
    match maze.get(x, y) {
        Some(TileType::Wall) => false,
        Some(TileType::GhostDoor) => state == GhostState::Eaten || in_house,
        Some(_) => true,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};

    const DT: f32 = 1.0 / 60.0;

    /// Open maze with border walls, ghost house in the center, and a ghost door.
    ///
    /// ```text
    /// W W W W W ... W
    /// W . . . . ... W
    /// W . . D . ... W   (D = GhostDoor at col 3, row 2)
    /// W . . H . ... W   (H = GhostHouse at col 3, row 3)
    /// W . . . . ... W
    /// ...
    /// ```
    fn ghost_maze() -> MazeData {
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
        // Ghost door and house
        tiles[2][3] = TileType::GhostDoor;
        tiles[3][3] = TileType::GhostHouse;
        MazeData { tiles }
    }

    /// Simple open maze for basic movement tests.
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

    /// Maze with a horizontal tunnel (open left and right edges on row 1).
    fn tunnel_maze() -> MazeData {
        let mut tiles = [[TileType::Wall; MAZE_WIDTH]; MAZE_HEIGHT];
        for x in 0..MAZE_WIDTH {
            tiles[1][x] = TileType::Empty;
        }
        MazeData { tiles }
    }

    // ── Construction ─────────────────────────────────────────

    #[test]
    fn new_places_at_grid_position() {
        let g = Ghost::new(GhostId::Blinky, 5, 10);
        assert_eq!(g.grid_x(), 5);
        assert_eq!(g.grid_y(), 10);
    }

    #[test]
    fn new_starts_in_scatter() {
        let g = Ghost::new(GhostId::Blinky, 1, 1);
        assert_eq!(g.state(), GhostState::Scatter);
    }

    #[test]
    fn new_starts_stationary() {
        let g = Ghost::new(GhostId::Pinky, 1, 1);
        assert_eq!(g.current_dir(), None);
        assert_eq!(g.queued_dir(), None);
    }

    #[test]
    fn new_starts_fully_arrived() {
        let g = Ghost::new(GhostId::Inky, 1, 1);
        assert!((g.move_progress() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn new_is_not_in_house() {
        let g = Ghost::new(GhostId::Clyde, 1, 1);
        assert!(!g.in_house());
    }

    #[test]
    fn new_in_house_flag() {
        let g = Ghost::new_in_house(GhostId::Pinky, 3, 3);
        assert!(g.in_house());
    }

    #[test]
    fn ghost_id_preserved() {
        let g = Ghost::new(GhostId::Inky, 1, 1);
        assert_eq!(g.id(), GhostId::Inky);
    }

    #[test]
    fn spawn_position_matches_initial() {
        let g = Ghost::new(GhostId::Blinky, 7, 9);
        assert_eq!(g.spawn_x(), 7);
        assert_eq!(g.spawn_y(), 9);
    }

    // ── Speed ────────────────────────────────────────────────

    #[test]
    fn default_speed_in_scatter() {
        let g = Ghost::new(GhostId::Blinky, 1, 1);
        assert!((g.speed() - NORMAL_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn speed_in_chase() {
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.enter_chase();
        assert!((g.speed() - NORMAL_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn speed_in_frightened() {
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.enter_frightened();
        assert!((g.speed() - FRIGHTENED_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn speed_in_eaten() {
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.enter_eaten();
        assert!((g.speed() - EATEN_SPEED).abs() < f32::EPSILON);
    }

    #[test]
    fn set_speeds_overrides_defaults() {
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.set_speeds(10.0, 5.0, 20.0);
        assert!((g.speed() - 10.0).abs() < f32::EPSILON);
        g.enter_frightened();
        assert!((g.speed() - 5.0).abs() < f32::EPSILON);
        g.enter_eaten();
        assert!((g.speed() - 20.0).abs() < f32::EPSILON);
    }

    // ── State transitions ────────────────────────────────────

    #[test]
    fn scatter_to_chase_reverses_direction() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        g.set_direction(Direction::Right);
        // Start moving
        g.update(DT, &maze);
        for _ in 0..3 {
            g.update(DT, &maze);
        }
        assert_eq!(g.current_dir(), Some(Direction::Right));

        g.enter_chase();
        assert_eq!(g.state(), GhostState::Chase);
        assert_eq!(g.current_dir(), Some(Direction::Left));
    }

    #[test]
    fn chase_to_scatter_reverses_direction() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        g.enter_chase(); // Scatter → Chase (reverses, but stationary so no visible effect)
        g.set_direction(Direction::Down);
        g.update(DT, &maze);
        for _ in 0..3 {
            g.update(DT, &maze);
        }
        assert_eq!(g.current_dir(), Some(Direction::Down));

        g.enter_scatter();
        assert_eq!(g.state(), GhostState::Scatter);
        assert_eq!(g.current_dir(), Some(Direction::Up));
    }

    #[test]
    fn enter_frightened_reverses_direction() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Pinky, 5, 5);
        g.set_direction(Direction::Right);
        g.update(DT, &maze);
        for _ in 0..3 {
            g.update(DT, &maze);
        }

        g.enter_frightened();
        assert_eq!(g.state(), GhostState::Frightened);
        assert_eq!(g.current_dir(), Some(Direction::Left));
    }

    #[test]
    fn frightened_to_chase_no_reversal() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        g.set_direction(Direction::Right);
        g.update(DT, &maze);
        for _ in 0..3 {
            g.update(DT, &maze);
        }
        g.enter_frightened();
        // Now frightened, direction is Left (reversed)
        assert_eq!(g.current_dir(), Some(Direction::Left));

        g.enter_chase();
        // Frightened → Chase: no reversal
        assert_eq!(g.current_dir(), Some(Direction::Left));
    }

    #[test]
    fn eaten_ignores_frightened() {
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.enter_eaten();
        g.enter_frightened();
        // Should still be eaten
        assert_eq!(g.state(), GhostState::Eaten);
    }

    #[test]
    fn enter_eaten_no_reversal() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        g.set_direction(Direction::Right);
        g.update(DT, &maze);
        for _ in 0..3 {
            g.update(DT, &maze);
        }

        g.enter_eaten();
        assert_eq!(g.state(), GhostState::Eaten);
        // Direction unchanged — no reversal on eaten
        assert_eq!(g.current_dir(), Some(Direction::Right));
    }

    // ── Respawn ──────────────────────────────────────────────

    #[test]
    fn respawn_resets_to_spawn_position() {
        let mut g = Ghost::new(GhostId::Blinky, 3, 3);
        g.enter_eaten();
        // Simulate movement away from spawn
        let maze = open_maze();
        g.set_direction(Direction::Right);
        for _ in 0..50 {
            g.update(DT, &maze);
        }
        assert!(g.grid_x() > 3);

        g.respawn();
        assert_eq!(g.grid_x(), 3);
        assert_eq!(g.grid_y(), 3);
        assert_eq!(g.prev_x(), 3);
        assert_eq!(g.prev_y(), 3);
    }

    #[test]
    fn respawn_sets_scatter_and_in_house() {
        let mut g = Ghost::new(GhostId::Blinky, 3, 3);
        g.enter_eaten();
        g.respawn();
        assert_eq!(g.state(), GhostState::Scatter);
        assert!(g.in_house());
    }

    #[test]
    fn respawn_clears_direction() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        g.set_direction(Direction::Right);
        for _ in 0..10 {
            g.update(DT, &maze);
        }
        g.respawn();
        assert_eq!(g.current_dir(), None);
        assert_eq!(g.queued_dir(), None);
        assert!((g.move_progress() - 1.0).abs() < f32::EPSILON);
    }

    // ── Basic movement ───────────────────────────────────────

    #[test]
    fn moves_right() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.set_direction(Direction::Right);
        for _ in 0..20 {
            g.update(DT, &maze);
        }
        assert!(g.grid_x() >= 2);
        assert_eq!(g.grid_y(), 1);
    }

    #[test]
    fn moves_down() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.set_direction(Direction::Down);
        for _ in 0..20 {
            g.update(DT, &maze);
        }
        assert_eq!(g.grid_x(), 1);
        assert!(g.grid_y() >= 2);
    }

    #[test]
    fn moves_left() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        g.set_direction(Direction::Left);
        for _ in 0..20 {
            g.update(DT, &maze);
        }
        assert!(g.grid_x() <= 4);
    }

    #[test]
    fn moves_up() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        g.set_direction(Direction::Up);
        for _ in 0..20 {
            g.update(DT, &maze);
        }
        assert!(g.grid_y() <= 4);
    }

    #[test]
    fn stationary_without_input() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        for _ in 0..100 {
            g.update(DT, &maze);
        }
        assert_eq!(g.grid_x(), 5);
        assert_eq!(g.grid_y(), 5);
    }

    // ── Wall collision ───────────────────────────────────────

    #[test]
    fn blocked_by_wall() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.set_direction(Direction::Left); // Wall at col 0
        for _ in 0..100 {
            g.update(DT, &maze);
        }
        assert_eq!(g.grid_x(), 1);
        assert_eq!(g.grid_y(), 1);
        assert_eq!(g.current_dir(), None);
    }

    #[test]
    fn continues_through_open_corridor() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.set_direction(Direction::Right);
        for _ in 0..500 {
            g.update(DT, &maze);
        }
        assert!(g.grid_x() > 3);
    }

    // ── Ghost house / door passability ───────────────────────

    #[test]
    fn ghost_walks_on_ghost_house_tiles() {
        let maze = ghost_maze();
        // Start at (3, 4) — one below ghost house, moving up into it.
        let mut g = Ghost::new(GhostId::Blinky, 3, 4);
        g.enter_eaten(); // Eaten so we can pass through door too
        g.set_direction(Direction::Up);
        for _ in 0..50 {
            g.update(DT, &maze);
        }
        // Should have entered ghost house tile at (3, 3)
        assert!(g.grid_y() <= 3);
    }

    #[test]
    fn normal_ghost_blocked_by_ghost_door() {
        let maze = ghost_maze();
        // Start at (3, 1) — above ghost door at (3, 2), moving down.
        let mut g = Ghost::new(GhostId::Blinky, 3, 1);
        // State is Scatter, not in house → door is blocked.
        g.set_direction(Direction::Down);
        for _ in 0..100 {
            g.update(DT, &maze);
        }
        // Should NOT pass through ghost door
        assert_eq!(g.grid_y(), 1);
    }

    #[test]
    fn eaten_ghost_passes_through_ghost_door() {
        let maze = ghost_maze();
        let mut g = Ghost::new(GhostId::Blinky, 3, 1);
        g.enter_eaten();
        g.set_direction(Direction::Down);
        for _ in 0..50 {
            g.update(DT, &maze);
        }
        // Should have passed through door at (3, 2) into house at (3, 3)
        assert!(g.grid_y() >= 2);
    }

    #[test]
    fn in_house_ghost_exits_through_door() {
        let maze = ghost_maze();
        // Ghost spawns inside house at (3, 3), in_house = true
        let mut g = Ghost::new_in_house(GhostId::Pinky, 3, 3);
        g.set_direction(Direction::Up); // Up through door at (3, 2) to (3, 1)
        for _ in 0..100 {
            g.update(DT, &maze);
        }
        // Should have exited through the door
        assert!(g.grid_y() <= 2);
        // After reaching a normal tile, in_house should be false
        assert!(!g.in_house());
    }

    #[test]
    fn ghost_cannot_reenter_after_exiting() {
        let maze = ghost_maze();
        let mut g = Ghost::new_in_house(GhostId::Pinky, 3, 3);
        g.set_direction(Direction::Up);
        // Exit the house
        for _ in 0..100 {
            g.update(DT, &maze);
        }
        assert!(!g.in_house());

        // Now try to go back down through the door
        g.set_direction(Direction::Down);
        for _ in 0..200 {
            g.update(DT, &maze);
        }
        // Should be blocked at row 1 (can't pass door at row 2)
        assert_eq!(g.grid_y(), 1);
    }

    #[test]
    fn in_house_becomes_true_on_ghost_house_tile() {
        let maze = ghost_maze();
        let mut g = Ghost::new(GhostId::Blinky, 3, 4);
        g.enter_eaten(); // Can pass through door
        g.set_direction(Direction::Up);
        // Move up to ghost house tile at (3, 3)
        for _ in 0..50 {
            g.update(DT, &maze);
        }
        if g.grid_y() == 3 && g.move_progress() >= 1.0 {
            assert!(g.in_house());
        }
    }

    // ── Tunnel wrapping ──────────────────────────────────────

    #[test]
    fn wraps_left_to_right() {
        let maze = tunnel_maze();
        let mut g = Ghost::new(GhostId::Blinky, 0, 1);
        g.set_direction(Direction::Left);
        for _ in 0..8 {
            g.update(DT, &maze);
        }
        assert_eq!(g.grid_x(), MAZE_WIDTH - 1);
    }

    #[test]
    fn wraps_right_to_left() {
        let maze = tunnel_maze();
        let mut g = Ghost::new(GhostId::Blinky, MAZE_WIDTH - 1, 1);
        g.set_direction(Direction::Right);
        for _ in 0..8 {
            g.update(DT, &maze);
        }
        assert_eq!(g.grid_x(), 0);
    }

    // ── World position / interpolation ───────────────────────

    #[test]
    fn world_position_at_grid_when_stationary() {
        let g = Ghost::new(GhostId::Blinky, 5, 10);
        let (wx, wz) = g.world_position(0.0);
        assert!((wx - 5.0).abs() < 1e-5);
        assert!((wz - 10.0).abs() < 1e-5);
    }

    #[test]
    fn world_position_interpolates_during_move() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 1, 1);
        g.set_direction(Direction::Right);
        g.update(DT, &maze);
        g.update(DT, &maze);
        let (wx, _) = g.world_position(0.0);
        assert!(wx > 1.0 && wx < 2.0, "wx = {}", wx);
    }

    #[test]
    fn world_position_at_full_progress_equals_grid() {
        let g = Ghost::new(GhostId::Blinky, 3, 7);
        let (wx, wz) = g.world_position(0.5);
        assert!((wx - 3.0).abs() < 1e-5);
        assert!((wz - 7.0).abs() < 1e-5);
    }

    // ── Mid-tile reversal ────────────────────────────────────

    #[test]
    fn reverse_mid_tile_via_set_direction() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        g.set_direction(Direction::Right);
        for _ in 0..2 {
            g.update(DT, &maze);
        }
        assert!(g.move_progress() < 1.0);

        g.set_direction(Direction::Left);
        g.update(DT, &maze);
        assert_eq!(g.current_dir(), Some(Direction::Left));
    }

    #[test]
    fn state_reversal_swaps_positions() {
        let maze = open_maze();
        let mut g = Ghost::new(GhostId::Blinky, 5, 5);
        g.set_direction(Direction::Right);
        g.update(DT, &maze);
        for _ in 0..3 {
            g.update(DT, &maze);
        }
        // Moving right: grid_x should be 6, prev_x should be 5
        let gx_before = g.grid_x();
        let px_before = g.prev_x();

        g.enter_frightened(); // reverses
        // After reversal mid-tile, positions should swap
        assert_eq!(g.grid_x(), px_before);
        assert_eq!(g.prev_x(), gx_before);
    }

    // ── Classic maze integration ─────────────────────────────

    #[test]
    fn ghost_house_exists_in_classic_maze() {
        let data = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/maze/classic.json"),
        )
        .expect("classic.json should exist");
        let maze = MazeData::from_json(&data).expect("should parse");
        let has_ghost_house = maze
            .tiles
            .iter()
            .any(|row| row.iter().any(|t| *t == TileType::GhostHouse));
        assert!(has_ghost_house, "classic maze should have a ghost house");
    }
}
