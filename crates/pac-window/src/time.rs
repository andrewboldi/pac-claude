use std::time::Instant;

/// Fixed timestep frequency for physics updates (60 Hz).
const FIXED_TIMESTEP_HZ: f64 = 60.0;

/// Duration of one fixed timestep in seconds.
const FIXED_DT: f64 = 1.0 / FIXED_TIMESTEP_HZ;

/// Maximum delta time clamp to prevent spiral-of-death when the app
/// freezes (e.g. window drag, debugger breakpoint). Any frame longer
/// than this is clamped so the accumulator doesn't queue hundreds of
/// physics ticks.
const MAX_DELTA: f64 = 0.25;

/// Tracks wall-clock timing for the game loop.
///
/// Provides both a variable `delta_time` (for rendering / interpolation)
/// and a fixed-timestep accumulator for deterministic physics updates.
///
/// # Usage
///
/// ```ignore
/// let mut time = TimeState::new();
///
/// // Each frame:
/// time.tick();
///
/// while time.should_do_fixed_update() {
///     physics_step(time.fixed_dt());
/// }
///
/// render(time.alpha());
/// ```
pub struct TimeState {
    /// Wall-clock instant of the previous tick.
    prev_instant: Instant,
    /// Seconds elapsed since the last tick (variable, clamped).
    delta_time: f64,
    /// Seconds elapsed since the first tick.
    total_time: f64,
    /// Number of completed ticks.
    frame_count: u64,
    /// Accumulated time not yet consumed by fixed updates.
    accumulator: f64,
}

impl TimeState {
    /// Create a new `TimeState` rooted at the current instant.
    pub fn new() -> Self {
        Self {
            prev_instant: Instant::now(),
            delta_time: 0.0,
            total_time: 0.0,
            frame_count: 0,
            accumulator: 0.0,
        }
    }

    /// Advance the clock. Call once at the top of each frame.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let raw_dt = now.duration_since(self.prev_instant).as_secs_f64();
        self.prev_instant = now;

        self.delta_time = raw_dt.min(MAX_DELTA);
        self.total_time += self.delta_time;
        self.frame_count += 1;
        self.accumulator += self.delta_time;
    }

    /// Returns `true` and drains one fixed step from the accumulator
    /// each time there is enough accumulated time.
    ///
    /// Call in a `while` loop to consume all pending fixed ticks:
    /// ```ignore
    /// while time.should_do_fixed_update() {
    ///     physics_step(time.fixed_dt());
    /// }
    /// ```
    pub fn should_do_fixed_update(&mut self) -> bool {
        if self.accumulator >= FIXED_DT {
            self.accumulator -= FIXED_DT;
            true
        } else {
            false
        }
    }

    /// The fixed timestep duration in seconds (1/60).
    pub fn fixed_dt(&self) -> f64 {
        FIXED_DT
    }

    /// Interpolation alpha for rendering between the last two fixed steps.
    ///
    /// Ranges from 0.0 (at the last fixed tick) to ~1.0 (just before the
    /// next one). Use this to interpolate visual positions so rendering
    /// stays smooth even when the physics rate differs from the frame rate.
    pub fn alpha(&self) -> f64 {
        self.accumulator / FIXED_DT
    }

    /// Seconds elapsed since the previous frame (variable, clamped to `MAX_DELTA`).
    pub fn delta_time(&self) -> f64 {
        self.delta_time
    }

    /// Total seconds elapsed since `TimeState::new()`.
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Number of completed frames (incremented each `tick()`).
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

impl Default for TimeState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn new_state_is_zeroed() {
        let t = TimeState::new();
        assert_eq!(t.delta_time(), 0.0);
        assert_eq!(t.total_time(), 0.0);
        assert_eq!(t.frame_count(), 0);
    }

    #[test]
    fn tick_advances_frame_count() {
        let mut t = TimeState::new();
        t.tick();
        assert_eq!(t.frame_count(), 1);
        t.tick();
        assert_eq!(t.frame_count(), 2);
    }

    #[test]
    fn delta_time_is_positive_after_tick() {
        let mut t = TimeState::new();
        thread::sleep(Duration::from_millis(5));
        t.tick();
        assert!(t.delta_time() > 0.0);
    }

    #[test]
    fn total_time_accumulates() {
        let mut t = TimeState::new();
        thread::sleep(Duration::from_millis(5));
        t.tick();
        let first = t.total_time();
        thread::sleep(Duration::from_millis(5));
        t.tick();
        assert!(t.total_time() > first);
    }

    #[test]
    fn fixed_update_drains_accumulator() {
        let mut t = TimeState::new();
        // Manually inject enough accumulated time for exactly 2 fixed steps.
        t.accumulator = FIXED_DT * 2.0 + 0.001;

        assert!(t.should_do_fixed_update());
        assert!(t.should_do_fixed_update());
        // Third call should fail — only ~0.001s remains.
        assert!(!t.should_do_fixed_update());
    }

    #[test]
    fn alpha_is_between_zero_and_one_after_partial_step() {
        let mut t = TimeState::new();
        t.accumulator = FIXED_DT * 0.5;
        let a = t.alpha();
        assert!(a > 0.0 && a < 1.0, "alpha was {}", a);
    }

    #[test]
    fn delta_clamped_to_max() {
        let mut t = TimeState::new();
        // Simulate a huge gap by resetting prev_instant far back.
        t.prev_instant = Instant::now() - Duration::from_secs(2);
        t.tick();
        assert!(
            t.delta_time() <= MAX_DELTA,
            "delta {} exceeded max {}",
            t.delta_time(),
            MAX_DELTA,
        );
    }

    #[test]
    fn fixed_dt_matches_constant() {
        let t = TimeState::new();
        assert!((t.fixed_dt() - 1.0 / 60.0).abs() < f64::EPSILON);
    }
}
