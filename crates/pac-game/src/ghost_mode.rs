//! Global ghost mode controller — scatter/chase cycle timer and frightened mode.
//!
//! The mode controller drives the global behavioral state for all ghosts.
//! Individual ghosts defer to [`Ghost::set_mode`](crate::ghost::Ghost::set_mode)
//! for side effects (speed change, direction reversal), but the *when* of mode
//! transitions is decided here.
//!
//! Classic Pac-Man level 1 schedule:
//! Scatter 7s → Chase 20s → Scatter 7s → Chase 20s → Scatter 5s → Chase 20s → Scatter 5s → Chase ∞
//!
//! Frightened mode pauses the scatter/chase timer and overrides all non-eaten
//! ghosts. When frightened expires, ghosts return to the current base mode.

use crate::ghost::{Ghost, GhostMode};

/// A single phase in the scatter/chase schedule.
#[derive(Debug, Clone, Copy)]
struct Phase {
    mode: GhostMode,
    duration: f32,
}

/// Default level-1 scatter/chase schedule (durations in seconds).
///
/// The final entry has `f32::INFINITY` — permanent chase.
const LEVEL1_SCHEDULE: &[Phase] = &[
    Phase { mode: GhostMode::Scatter, duration: 7.0 },
    Phase { mode: GhostMode::Chase, duration: 20.0 },
    Phase { mode: GhostMode::Scatter, duration: 7.0 },
    Phase { mode: GhostMode::Chase, duration: 20.0 },
    Phase { mode: GhostMode::Scatter, duration: 5.0 },
    Phase { mode: GhostMode::Chase, duration: 20.0 },
    Phase { mode: GhostMode::Scatter, duration: 5.0 },
    Phase { mode: GhostMode::Chase, duration: f32::INFINITY },
];

/// Duration of frightened mode in seconds (level 1).
const FRIGHTENED_DURATION: f32 = 6.0;

/// How many seconds before frightened ends the flashing warning begins.
const FLASH_WARNING_TIME: f32 = 2.0;

/// Global ghost mode controller.
///
/// Manages the scatter/chase cycle and frightened-mode overlay. Call
/// [`update`](GhostModeController::update) each fixed-timestep tick, and
/// [`trigger_frightened`](GhostModeController::trigger_frightened) when a
/// power pellet is collected.
pub struct GhostModeController {
    /// The scatter/chase schedule.
    schedule: &'static [Phase],
    /// Current index into `schedule`.
    phase_index: usize,
    /// Time elapsed in the current scatter/chase phase.
    phase_elapsed: f32,
    /// The active base mode (Scatter or Chase) from the schedule.
    base_mode: GhostMode,

    /// Whether frightened mode is currently active.
    frightened_active: bool,
    /// Time remaining for frightened mode (counts down).
    frightened_timer: f32,
    /// Total frightened duration for the current activation.
    frightened_duration: f32,
    /// Number of ghosts eaten during the current frightened period (for scoring).
    ghosts_eaten_count: u32,
}

impl GhostModeController {
    /// Create a new controller with the default level-1 schedule.
    ///
    /// Starts in the first phase (Scatter, 7 seconds).
    pub fn new() -> Self {
        Self {
            schedule: LEVEL1_SCHEDULE,
            phase_index: 0,
            phase_elapsed: 0.0,
            base_mode: LEVEL1_SCHEDULE[0].mode,
            frightened_active: false,
            frightened_timer: 0.0,
            frightened_duration: FRIGHTENED_DURATION,
            ghosts_eaten_count: 0,
        }
    }

    // ── Accessors ─────────────────────────────────────────────

    /// The current base mode from the scatter/chase schedule.
    pub fn base_mode(&self) -> GhostMode {
        self.base_mode
    }

    /// Whether frightened mode is currently active.
    pub fn is_frightened(&self) -> bool {
        self.frightened_active
    }

    /// Whether the frightened timer is in the flashing warning phase.
    ///
    /// Returns `true` when frightened is active and remaining time is within
    /// the flash warning window. Used by the renderer to flash ghost sprites.
    pub fn is_flashing(&self) -> bool {
        self.frightened_active && self.frightened_timer <= FLASH_WARNING_TIME
    }

    /// Time remaining in the current frightened period (0.0 if not frightened).
    pub fn frightened_time_remaining(&self) -> f32 {
        if self.frightened_active {
            self.frightened_timer
        } else {
            0.0
        }
    }

    /// Number of ghosts eaten during the current frightened period.
    ///
    /// Classic scoring: 200 × 2^(count-1) → 200, 400, 800, 1600.
    /// Resets to 0 when frightened ends or a new frightened period starts.
    pub fn ghosts_eaten_count(&self) -> u32 {
        self.ghosts_eaten_count
    }

    /// The effective mode that non-eaten ghosts should be in right now.
    ///
    /// Returns `Frightened` if frightened is active, otherwise the base mode.
    pub fn effective_mode(&self) -> GhostMode {
        if self.frightened_active {
            GhostMode::Frightened
        } else {
            self.base_mode
        }
    }

    /// Current phase index in the scatter/chase schedule.
    pub fn phase_index(&self) -> usize {
        self.phase_index
    }

    /// Time elapsed in the current scatter/chase phase.
    pub fn phase_elapsed(&self) -> f32 {
        self.phase_elapsed
    }

    // ── Update ────────────────────────────────────────────────

    /// Advance timers by `dt` seconds and apply mode transitions to ghosts.
    ///
    /// Call once per fixed-timestep tick. When a phase boundary is crossed or
    /// frightened expires, ghosts are transitioned via [`Ghost::set_mode`].
    pub fn update(&mut self, dt: f32, ghosts: &mut [Ghost]) {
        if self.frightened_active {
            self.update_frightened(dt, ghosts);
        } else {
            self.update_scatter_chase(dt, ghosts);
        }
    }

    /// Activate frightened mode on all non-eaten ghosts.
    ///
    /// Call when a power pellet is collected. If already frightened, the timer
    /// resets (classic Pac-Man behavior). The scatter/chase timer is paused
    /// for the duration.
    pub fn trigger_frightened(&mut self, ghosts: &mut [Ghost]) {
        self.frightened_active = true;
        self.frightened_timer = self.frightened_duration;
        self.ghosts_eaten_count = 0;

        for ghost in ghosts.iter_mut() {
            if ghost.mode() != GhostMode::Eaten {
                ghost.set_mode(GhostMode::Frightened);
            }
        }
    }

    /// Notify the controller that a ghost was eaten (by index).
    ///
    /// Increments the eaten counter for scoring. The caller is responsible for
    /// calling [`Ghost::set_mode(Eaten)`](Ghost::set_mode) on the ghost itself.
    pub fn notify_ghost_eaten(&mut self) {
        self.ghosts_eaten_count += 1;
    }

    /// Score value for the most recently eaten ghost.
    ///
    /// Classic scoring: 200 × 2^(count-1) → 200, 400, 800, 1600.
    /// Returns 0 if no ghosts have been eaten this frightened period.
    pub fn last_eat_score(&self) -> u32 {
        if self.ghosts_eaten_count == 0 {
            return 0;
        }
        200 * (1 << (self.ghosts_eaten_count - 1))
    }

    /// Transition an eaten ghost back to the current base mode.
    ///
    /// Call when an eaten ghost has reached its home tile (see
    /// [`Ghost::reached_home`]). The ghost re-enters the scatter/chase cycle.
    pub fn return_ghost_to_base(&self, ghost: &mut Ghost) {
        ghost.set_mode(self.base_mode);
    }

    /// Reset the controller to its initial state.
    ///
    /// Used when starting a new life or a new level.
    pub fn reset(&mut self) {
        self.phase_index = 0;
        self.phase_elapsed = 0.0;
        self.base_mode = self.schedule[0].mode;
        self.frightened_active = false;
        self.frightened_timer = 0.0;
        self.ghosts_eaten_count = 0;
    }

    // ── Internal ──────────────────────────────────────────────

    /// Tick the scatter/chase timer and transition phases when due.
    fn update_scatter_chase(&mut self, dt: f32, ghosts: &mut [Ghost]) {
        let phase = &self.schedule[self.phase_index];

        self.phase_elapsed += dt;

        // Check for phase transition.
        if self.phase_elapsed >= phase.duration && self.phase_index + 1 < self.schedule.len() {
            let overflow = self.phase_elapsed - phase.duration;
            self.phase_index += 1;
            self.phase_elapsed = overflow;
            self.base_mode = self.schedule[self.phase_index].mode;

            // Apply new base mode to all non-eaten ghosts.
            for ghost in ghosts.iter_mut() {
                if ghost.mode() != GhostMode::Eaten {
                    ghost.set_mode(self.base_mode);
                }
            }
        }
    }

    /// Tick the frightened timer and end frightened mode when expired.
    fn update_frightened(&mut self, dt: f32, ghosts: &mut [Ghost]) {
        self.frightened_timer -= dt;

        if self.frightened_timer <= 0.0 {
            self.frightened_active = false;
            self.frightened_timer = 0.0;
            self.ghosts_eaten_count = 0;

            // Return all frightened ghosts to the base mode.
            for ghost in ghosts.iter_mut() {
                if ghost.mode() == GhostMode::Frightened {
                    ghost.set_mode(self.base_mode);
                }
            }
        }
    }
}

impl Default for GhostModeController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ghost::{Ghost, GhostMode};
    use crate::maze::{MazeData, TileType, MAZE_HEIGHT, MAZE_WIDTH};

    const DT: f32 = 1.0 / 60.0;

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

    fn four_ghosts() -> Vec<Ghost> {
        vec![
            Ghost::new(5, 5),
            Ghost::new(6, 5),
            Ghost::new(7, 5),
            Ghost::new(8, 5),
        ]
    }

    /// Advance `n` seconds by ticking at 60 Hz.
    fn advance(ctrl: &mut GhostModeController, ghosts: &mut [Ghost], seconds: f32) {
        let ticks = (seconds / DT).round() as usize;
        for _ in 0..ticks {
            ctrl.update(DT, ghosts);
        }
    }

    // ── Construction ──────────────────────────────────────────

    #[test]
    fn starts_in_scatter() {
        let ctrl = GhostModeController::new();
        assert_eq!(ctrl.base_mode(), GhostMode::Scatter);
    }

    #[test]
    fn starts_not_frightened() {
        let ctrl = GhostModeController::new();
        assert!(!ctrl.is_frightened());
        assert!(!ctrl.is_flashing());
    }

    #[test]
    fn effective_mode_is_scatter_initially() {
        let ctrl = GhostModeController::new();
        assert_eq!(ctrl.effective_mode(), GhostMode::Scatter);
    }

    #[test]
    fn starts_at_phase_zero() {
        let ctrl = GhostModeController::new();
        assert_eq!(ctrl.phase_index(), 0);
        assert!((ctrl.phase_elapsed() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ghosts_eaten_count_starts_at_zero() {
        let ctrl = GhostModeController::new();
        assert_eq!(ctrl.ghosts_eaten_count(), 0);
    }

    #[test]
    fn default_same_as_new() {
        let a = GhostModeController::new();
        let b = GhostModeController::default();
        assert_eq!(a.base_mode(), b.base_mode());
        assert_eq!(a.phase_index(), b.phase_index());
        assert!(!a.is_frightened());
        assert!(!b.is_frightened());
    }

    // ── Scatter/chase cycle ───────────────────────────────────

    #[test]
    fn transitions_to_chase_after_7_seconds() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        advance(&mut ctrl, &mut ghosts, 7.0);
        assert_eq!(ctrl.base_mode(), GhostMode::Chase);
        assert_eq!(ctrl.phase_index(), 1);
    }

    #[test]
    fn ghosts_receive_chase_mode_at_transition() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        advance(&mut ctrl, &mut ghosts, 7.0);
        for ghost in &ghosts {
            assert_eq!(ghost.mode(), GhostMode::Chase);
        }
    }

    #[test]
    fn stays_scatter_before_7_seconds() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        advance(&mut ctrl, &mut ghosts, 6.0);
        assert_eq!(ctrl.base_mode(), GhostMode::Scatter);
        assert_eq!(ctrl.phase_index(), 0);
    }

    #[test]
    fn transitions_back_to_scatter_after_27_seconds() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        // 7s scatter + 20s chase = 27s → back to scatter
        // Use 27.1 to avoid floating-point accumulation landing exactly on boundary.
        advance(&mut ctrl, &mut ghosts, 27.1);
        assert_eq!(ctrl.base_mode(), GhostMode::Scatter);
        assert_eq!(ctrl.phase_index(), 2);
    }

    #[test]
    fn full_schedule_reaches_permanent_chase() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        // 7+20+7+20+5+20+5 = 84 seconds to reach permanent chase.
        // Use 84.5 to avoid floating-point accumulation landing on boundary.
        advance(&mut ctrl, &mut ghosts, 84.5);
        assert_eq!(ctrl.base_mode(), GhostMode::Chase);
        assert_eq!(ctrl.phase_index(), 7);
    }

    #[test]
    fn permanent_chase_stays_forever() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        advance(&mut ctrl, &mut ghosts, 84.5);
        assert_eq!(ctrl.base_mode(), GhostMode::Chase);
        // Advance much more — should stay in chase
        advance(&mut ctrl, &mut ghosts, 300.0);
        assert_eq!(ctrl.base_mode(), GhostMode::Chase);
        assert_eq!(ctrl.phase_index(), 7);
    }

    #[test]
    fn eaten_ghost_skips_scatter_chase_transition() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        // Put ghost 0 into Eaten mode
        ghosts[0].set_mode(GhostMode::Chase);
        ghosts[0].set_mode(GhostMode::Eaten);

        // Transition to chase at 7s
        advance(&mut ctrl, &mut ghosts, 7.0);

        // Ghost 0 should still be Eaten
        assert_eq!(ghosts[0].mode(), GhostMode::Eaten);
        // Others should be Chase
        assert_eq!(ghosts[1].mode(), GhostMode::Chase);
        assert_eq!(ghosts[2].mode(), GhostMode::Chase);
        assert_eq!(ghosts[3].mode(), GhostMode::Chase);
    }

    // ── Frightened mode ───────────────────────────────────────

    #[test]
    fn trigger_frightened_activates() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        assert!(ctrl.is_frightened());
    }

    #[test]
    fn trigger_frightened_sets_ghost_modes() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        for ghost in &ghosts {
            assert_eq!(ghost.mode(), GhostMode::Frightened);
        }
    }

    #[test]
    fn effective_mode_is_frightened_when_active() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        assert_eq!(ctrl.effective_mode(), GhostMode::Frightened);
    }

    #[test]
    fn frightened_does_not_affect_eaten_ghost() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ghosts[0].set_mode(GhostMode::Chase);
        ghosts[0].set_mode(GhostMode::Eaten);

        ctrl.trigger_frightened(&mut ghosts);
        assert_eq!(ghosts[0].mode(), GhostMode::Eaten);
        assert_eq!(ghosts[1].mode(), GhostMode::Frightened);
    }

    #[test]
    fn frightened_expires_after_duration() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        advance(&mut ctrl, &mut ghosts, FRIGHTENED_DURATION);
        assert!(!ctrl.is_frightened());
    }

    #[test]
    fn ghosts_return_to_base_mode_after_frightened() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        // Base mode is Scatter
        ctrl.trigger_frightened(&mut ghosts);
        advance(&mut ctrl, &mut ghosts, FRIGHTENED_DURATION);
        for ghost in &ghosts {
            assert_eq!(ghost.mode(), GhostMode::Scatter);
        }
    }

    #[test]
    fn frightened_pauses_scatter_chase_timer() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        // Advance 5 seconds into scatter
        advance(&mut ctrl, &mut ghosts, 5.0);
        assert_eq!(ctrl.base_mode(), GhostMode::Scatter);

        // Trigger frightened
        ctrl.trigger_frightened(&mut ghosts);
        let elapsed_before = ctrl.phase_elapsed();

        // Advance through frightened duration
        advance(&mut ctrl, &mut ghosts, FRIGHTENED_DURATION);

        // Scatter/chase timer should not have advanced during frightened
        assert!((ctrl.phase_elapsed() - elapsed_before).abs() < DT * 2.0);
    }

    #[test]
    fn scatter_chase_resumes_after_frightened() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        // Advance 5 seconds into scatter (2s remaining)
        advance(&mut ctrl, &mut ghosts, 5.0);

        // Trigger and complete frightened
        ctrl.trigger_frightened(&mut ghosts);
        advance(&mut ctrl, &mut ghosts, FRIGHTENED_DURATION);

        // Now 2 more seconds should transition to chase
        advance(&mut ctrl, &mut ghosts, 2.0);
        assert_eq!(ctrl.base_mode(), GhostMode::Chase);
    }

    #[test]
    fn double_power_pellet_resets_frightened_timer() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        advance(&mut ctrl, &mut ghosts, 3.0); // 3s into frightened

        // Second power pellet
        ctrl.trigger_frightened(&mut ghosts);
        assert!(ctrl.is_frightened());
        // Timer should be reset — need full duration again
        assert!(ctrl.frightened_time_remaining() > FRIGHTENED_DURATION - DT);
    }

    #[test]
    fn double_power_pellet_resets_eat_counter() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        ctrl.notify_ghost_eaten();
        ctrl.notify_ghost_eaten();
        assert_eq!(ctrl.ghosts_eaten_count(), 2);

        ctrl.trigger_frightened(&mut ghosts);
        assert_eq!(ctrl.ghosts_eaten_count(), 0);
    }

    // ── Flashing ──────────────────────────────────────────────

    #[test]
    fn not_flashing_at_start_of_frightened() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        assert!(!ctrl.is_flashing());
    }

    #[test]
    fn flashing_near_end_of_frightened() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        // Advance to within the flash warning window
        advance(&mut ctrl, &mut ghosts, FRIGHTENED_DURATION - FLASH_WARNING_TIME);
        assert!(ctrl.is_flashing());
    }

    #[test]
    fn not_flashing_when_not_frightened() {
        let ctrl = GhostModeController::new();
        assert!(!ctrl.is_flashing());
    }

    // ── Ghost eaten scoring ───────────────────────────────────

    #[test]
    fn first_eat_scores_200() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        ctrl.notify_ghost_eaten();
        assert_eq!(ctrl.last_eat_score(), 200);
    }

    #[test]
    fn second_eat_scores_400() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        ctrl.notify_ghost_eaten();
        ctrl.notify_ghost_eaten();
        assert_eq!(ctrl.last_eat_score(), 400);
    }

    #[test]
    fn third_eat_scores_800() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        ctrl.notify_ghost_eaten();
        ctrl.notify_ghost_eaten();
        ctrl.notify_ghost_eaten();
        assert_eq!(ctrl.last_eat_score(), 800);
    }

    #[test]
    fn fourth_eat_scores_1600() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        ctrl.notify_ghost_eaten();
        ctrl.notify_ghost_eaten();
        ctrl.notify_ghost_eaten();
        ctrl.notify_ghost_eaten();
        assert_eq!(ctrl.last_eat_score(), 1600);
    }

    #[test]
    fn no_eats_scores_zero() {
        let ctrl = GhostModeController::new();
        assert_eq!(ctrl.last_eat_score(), 0);
    }

    #[test]
    fn eat_counter_resets_when_frightened_ends() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        ctrl.notify_ghost_eaten();
        ctrl.notify_ghost_eaten();
        advance(&mut ctrl, &mut ghosts, FRIGHTENED_DURATION);
        assert_eq!(ctrl.ghosts_eaten_count(), 0);
    }

    // ── Return ghost to base ──────────────────────────────────

    #[test]
    fn return_ghost_to_base_during_scatter() {
        let ctrl = GhostModeController::new();
        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        ghost.set_mode(GhostMode::Eaten);

        ctrl.return_ghost_to_base(&mut ghost);
        assert_eq!(ghost.mode(), GhostMode::Scatter);
    }

    #[test]
    fn return_ghost_to_base_during_chase() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        advance(&mut ctrl, &mut ghosts, 7.0); // transition to chase

        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        ghost.set_mode(GhostMode::Eaten);

        ctrl.return_ghost_to_base(&mut ghost);
        assert_eq!(ghost.mode(), GhostMode::Chase);
    }

    #[test]
    fn return_ghost_to_base_during_frightened_uses_base_not_frightened() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);

        let mut ghost = Ghost::new(5, 5);
        ghost.set_mode(GhostMode::Chase);
        ghost.set_mode(GhostMode::Eaten);

        // Should return to scatter (the base), not frightened
        ctrl.return_ghost_to_base(&mut ghost);
        assert_eq!(ghost.mode(), GhostMode::Scatter);
    }

    // ── Reset ─────────────────────────────────────────────────

    #[test]
    fn reset_restores_initial_state() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        advance(&mut ctrl, &mut ghosts, 30.0);
        ctrl.trigger_frightened(&mut ghosts);
        ctrl.notify_ghost_eaten();

        ctrl.reset();
        assert_eq!(ctrl.base_mode(), GhostMode::Scatter);
        assert_eq!(ctrl.phase_index(), 0);
        assert!((ctrl.phase_elapsed() - 0.0).abs() < f32::EPSILON);
        assert!(!ctrl.is_frightened());
        assert_eq!(ctrl.ghosts_eaten_count(), 0);
    }

    // ── Frightened time remaining ─────────────────────────────

    #[test]
    fn frightened_time_remaining_zero_when_not_frightened() {
        let ctrl = GhostModeController::new();
        assert!((ctrl.frightened_time_remaining() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn frightened_time_remaining_decreases() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        ctrl.trigger_frightened(&mut ghosts);
        let t0 = ctrl.frightened_time_remaining();
        advance(&mut ctrl, &mut ghosts, 2.0);
        let t1 = ctrl.frightened_time_remaining();
        assert!(t1 < t0);
        assert!((t0 - t1 - 2.0).abs() < DT * 2.0);
    }

    // ── Integration: full sequence ────────────────────────────

    #[test]
    fn full_game_sequence() {
        let mut ctrl = GhostModeController::new();
        let mut ghosts = four_ghosts();
        let maze = open_maze();

        // Start in scatter
        assert_eq!(ctrl.base_mode(), GhostMode::Scatter);

        // 3 seconds in, power pellet
        advance(&mut ctrl, &mut ghosts, 3.0);
        ctrl.trigger_frightened(&mut ghosts);
        assert!(ctrl.is_frightened());

        // Eat two ghosts during frightened
        ctrl.notify_ghost_eaten();
        ghosts[0].set_mode(GhostMode::Eaten);
        ctrl.notify_ghost_eaten();
        ghosts[1].set_mode(GhostMode::Eaten);

        // Advance past frightened
        advance(&mut ctrl, &mut ghosts, FRIGHTENED_DURATION);
        assert!(!ctrl.is_frightened());

        // Ghosts 2,3 should be back to scatter; 0,1 still eaten
        assert_eq!(ghosts[2].mode(), GhostMode::Scatter);
        assert_eq!(ghosts[3].mode(), GhostMode::Scatter);
        assert_eq!(ghosts[0].mode(), GhostMode::Eaten);
        assert_eq!(ghosts[1].mode(), GhostMode::Eaten);

        // Return eaten ghosts
        ctrl.return_ghost_to_base(&mut ghosts[0]);
        ctrl.return_ghost_to_base(&mut ghosts[1]);
        assert_eq!(ghosts[0].mode(), GhostMode::Scatter);
        assert_eq!(ghosts[1].mode(), GhostMode::Scatter);

        // Continue to chase transition (4 more seconds since we had 3s + 6s frightened paused at 3s)
        // scatter timer was at 3s, 4 more = 7s total → chase
        advance(&mut ctrl, &mut ghosts, 4.0);
        assert_eq!(ctrl.base_mode(), GhostMode::Chase);

        // Give ghosts directions so they can move
        for ghost in ghosts.iter_mut() {
            ghost.set_direction(crate::pacman::Direction::Right);
            ghost.update(DT, &maze);
        }
    }
}
