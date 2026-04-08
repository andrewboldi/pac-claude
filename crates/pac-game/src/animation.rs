//! Animation controllers — mouth cycle, ghost float, death spiral, frightened flashing.
//!
//! Each controller is a pure time-driven state machine that outputs values for
//! the renderer. They have no rendering dependencies and are advanced via
//! `update(dt)` each physics tick.

use std::f32::consts::{PI, TAU};

// ── Mouth cycle ──────────────────────────────────────────────────

/// Pac-Man's opening-and-closing mouth animation.
///
/// Produces a `mouth_angle` in radians (0 = closed, `max_angle` = fully open)
/// that oscillates sinusoidally at a configurable frequency. The animation
/// pauses automatically when Pac-Man is stationary.
pub struct MouthCycle {
    /// Maximum opening half-angle in radians (~45° by default).
    max_angle: f32,
    /// Full open→close cycles per second.
    frequency: f32,
    /// Accumulated phase in radians (wraps at TAU).
    phase: f32,
    /// Whether Pac-Man is currently moving (mouth animates only while moving).
    moving: bool,
}

/// Default maximum mouth half-angle (~45°).
const DEFAULT_MOUTH_ANGLE: f32 = PI / 4.0;
/// Default mouth cycle speed — 8 open/close cycles per second.
const DEFAULT_MOUTH_FREQ: f32 = 8.0;

impl MouthCycle {
    /// Create a new mouth animation with default parameters.
    pub fn new() -> Self {
        Self {
            max_angle: DEFAULT_MOUTH_ANGLE,
            frequency: DEFAULT_MOUTH_FREQ,
            phase: 0.0,
            moving: false,
        }
    }

    /// Set the maximum opening half-angle in radians.
    pub fn set_max_angle(&mut self, angle: f32) {
        self.max_angle = angle;
    }

    /// Set the cycle frequency in Hz.
    pub fn set_frequency(&mut self, freq: f32) {
        self.frequency = freq;
    }

    /// Tell the animation whether Pac-Man is moving. The mouth only
    /// animates while moving; it freezes at its current angle when stopped.
    pub fn set_moving(&mut self, moving: bool) {
        self.moving = moving;
    }

    /// Advance the animation by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        if self.moving {
            self.phase += TAU * self.frequency * dt;
            if self.phase >= TAU {
                self.phase -= TAU;
            }
        }
    }

    /// Current mouth half-angle in radians (0 = closed, `max_angle` = open).
    ///
    /// Uses the absolute value of a sine wave so the mouth smoothly opens
    /// and closes without snapping at the zero crossing.
    pub fn angle(&self) -> f32 {
        self.max_angle * self.phase.sin().abs()
    }

    /// Normalized opening amount, 0.0 (closed) to 1.0 (fully open).
    pub fn openness(&self) -> f32 {
        self.phase.sin().abs()
    }
}

impl Default for MouthCycle {
    fn default() -> Self {
        Self::new()
    }
}

// ── Ghost float ──────────────────────────────────────────────────

/// Sinusoidal vertical bob applied to ghost sprites.
///
/// Produces a smooth `offset` that oscillates between `−amplitude` and
/// `+amplitude`, giving ghosts a hovering look.
pub struct GhostFloat {
    /// Peak displacement (world-space units).
    amplitude: f32,
    /// Oscillation frequency in Hz.
    frequency: f32,
    /// Accumulated phase in radians.
    phase: f32,
}

/// Default bob amplitude in world-space units (0.06 ≈ subtle).
const DEFAULT_FLOAT_AMPLITUDE: f32 = 0.06;
/// Default bob frequency — 2 Hz.
const DEFAULT_FLOAT_FREQ: f32 = 2.0;

impl GhostFloat {
    /// Create a new float animation with default parameters.
    pub fn new() -> Self {
        Self {
            amplitude: DEFAULT_FLOAT_AMPLITUDE,
            frequency: DEFAULT_FLOAT_FREQ,
            phase: 0.0,
        }
    }

    /// Create with a custom starting phase (useful to desynchronize ghosts).
    pub fn with_phase(phase: f32) -> Self {
        Self { phase, ..Self::new() }
    }

    /// Set the bob amplitude.
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude;
    }

    /// Set the oscillation frequency in Hz.
    pub fn set_frequency(&mut self, freq: f32) {
        self.frequency = freq;
    }

    /// Advance the animation by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        self.phase += TAU * self.frequency * dt;
        if self.phase >= TAU {
            self.phase -= TAU;
        }
    }

    /// Current vertical offset in world-space units.
    pub fn offset(&self) -> f32 {
        self.amplitude * self.phase.sin()
    }

    /// Current phase in radians.
    pub fn phase(&self) -> f32 {
        self.phase
    }
}

impl Default for GhostFloat {
    fn default() -> Self {
        Self::new()
    }
}

// ── Death spiral ─────────────────────────────────────────────────

/// Pac-Man death animation — the sprite rotates and collapses.
///
/// This is a one-shot animation that runs for `duration` seconds. While
/// active, `rotation` increases (the wedge spins) and `collapse` grows
/// from 0 to 1 (the opening angle widens until the sprite disappears).
///
/// Call [`start`](DeathSpiral::start) to begin, then [`update`](DeathSpiral::update)
/// each tick. Check [`finished`](DeathSpiral::finished) to know when it's done.
pub struct DeathSpiral {
    /// Total animation duration in seconds.
    duration: f32,
    /// Elapsed time since start.
    elapsed: f32,
    /// Whether the animation is currently playing.
    active: bool,
}

/// Default death animation duration — 1.5 seconds (matches classic Pac-Man).
const DEFAULT_DEATH_DURATION: f32 = 1.5;

/// Number of full rotations during the death animation.
const DEATH_ROTATIONS: f32 = 1.5;

impl DeathSpiral {
    /// Create a new (inactive) death spiral animation.
    pub fn new() -> Self {
        Self {
            duration: DEFAULT_DEATH_DURATION,
            elapsed: 0.0,
            active: false,
        }
    }

    /// Set the animation duration in seconds.
    pub fn set_duration(&mut self, duration: f32) {
        self.duration = duration;
    }

    /// Begin the death animation from the start.
    pub fn start(&mut self) {
        self.elapsed = 0.0;
        self.active = true;
    }

    /// Reset to inactive state.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.active = false;
    }

    /// Advance by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        if self.active {
            self.elapsed += dt;
            if self.elapsed >= self.duration {
                self.elapsed = self.duration;
                self.active = false;
            }
        }
    }

    /// Animation progress, 0.0 (start) to 1.0 (finished).
    pub fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        (self.elapsed / self.duration).clamp(0.0, 1.0)
    }

    /// Current rotation angle in radians (increases as the sprite spins).
    pub fn rotation(&self) -> f32 {
        self.progress() * DEATH_ROTATIONS * TAU
    }

    /// Collapse amount, 0.0 (fully visible) to 1.0 (fully collapsed).
    ///
    /// Uses an ease-in curve so the collapse accelerates at the end.
    pub fn collapse(&self) -> f32 {
        let t = self.progress();
        t * t
    }

    /// Whether the animation is currently playing.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Whether the animation has finished (ran to completion).
    pub fn finished(&self) -> bool {
        !self.active && self.elapsed >= self.duration
    }
}

impl Default for DeathSpiral {
    fn default() -> Self {
        Self::new()
    }
}

// ── Frightened flashing ──────────────────────────────────────────

/// Frightened-mode ghost flashing — alternates between blue and white
/// as the power pellet timer nears expiry.
///
/// Flashing activates when the remaining frightened time drops below a
/// threshold. The flash rate increases as the timer decreases, warning
/// the player that ghosts are about to recover.
pub struct FrightenedFlash {
    /// Duration (seconds) before frightened ends when flashing begins.
    warn_threshold: f32,
    /// Base flash frequency in Hz when flashing first starts.
    base_frequency: f32,
    /// Maximum flash frequency in Hz as the timer approaches zero.
    max_frequency: f32,
    /// Accumulated phase.
    phase: f32,
}

/// Start flashing with this many seconds remaining on the frightened timer.
const DEFAULT_WARN_THRESHOLD: f32 = 3.0;
/// Initial flash rate.
const DEFAULT_FLASH_BASE_FREQ: f32 = 3.0;
/// Final flash rate as timer hits zero.
const DEFAULT_FLASH_MAX_FREQ: f32 = 8.0;

impl FrightenedFlash {
    /// Create a new flash controller with default parameters.
    pub fn new() -> Self {
        Self {
            warn_threshold: DEFAULT_WARN_THRESHOLD,
            base_frequency: DEFAULT_FLASH_BASE_FREQ,
            max_frequency: DEFAULT_FLASH_MAX_FREQ,
            phase: 0.0,
        }
    }

    /// Set the warning threshold in seconds.
    pub fn set_warn_threshold(&mut self, seconds: f32) {
        self.warn_threshold = seconds;
    }

    /// Advance by `dt` seconds given the `remaining` frightened time.
    ///
    /// Flashing only occurs when `remaining` is below the warn threshold.
    /// The flash rate interpolates from `base_frequency` (at the threshold)
    /// to `max_frequency` (at zero).
    pub fn update(&mut self, dt: f32, remaining: f32) {
        if remaining > 0.0 && remaining <= self.warn_threshold {
            let urgency = 1.0 - (remaining / self.warn_threshold);
            let freq = self.base_frequency + (self.max_frequency - self.base_frequency) * urgency;
            self.phase += TAU * freq * dt;
            if self.phase >= TAU {
                self.phase -= TAU;
            }
        } else {
            self.phase = 0.0;
        }
    }

    /// Whether flashing should be active (remaining time is in the warn window).
    pub fn should_flash(&self, remaining: f32) -> bool {
        remaining > 0.0 && remaining <= self.warn_threshold
    }

    /// Whether the ghost should show the white (alternate) sprite right now.
    ///
    /// Returns `false` when not flashing or when in the blue half of the cycle.
    pub fn is_white(&self) -> bool {
        self.phase.sin() > 0.0
    }

    /// Reset the phase accumulator (call when entering frightened mode).
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Warning threshold in seconds.
    pub fn warn_threshold(&self) -> f32 {
        self.warn_threshold
    }
}

impl Default for FrightenedFlash {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    // ── MouthCycle ───────────────────────────────────────────

    #[test]
    fn mouth_starts_closed() {
        let mouth = MouthCycle::new();
        assert!(mouth.angle().abs() < 1e-5, "angle = {}", mouth.angle());
    }

    #[test]
    fn mouth_does_not_animate_when_stationary() {
        let mut mouth = MouthCycle::new();
        for _ in 0..100 {
            mouth.update(DT);
        }
        assert!(mouth.angle().abs() < 1e-5);
    }

    #[test]
    fn mouth_animates_when_moving() {
        let mut mouth = MouthCycle::new();
        mouth.set_moving(true);
        // Advance enough to see a non-zero angle.
        for _ in 0..5 {
            mouth.update(DT);
        }
        assert!(mouth.angle() > 0.0, "angle = {}", mouth.angle());
    }

    #[test]
    fn mouth_angle_within_bounds() {
        let mut mouth = MouthCycle::new();
        mouth.set_moving(true);
        for _ in 0..1000 {
            mouth.update(DT);
            assert!(mouth.angle() >= 0.0);
            assert!(
                mouth.angle() <= mouth.max_angle + 1e-5,
                "angle {} > max {}",
                mouth.angle(),
                mouth.max_angle
            );
        }
    }

    #[test]
    fn mouth_openness_within_unit_range() {
        let mut mouth = MouthCycle::new();
        mouth.set_moving(true);
        for _ in 0..1000 {
            mouth.update(DT);
            let o = mouth.openness();
            assert!(o >= 0.0 && o <= 1.0 + 1e-5, "openness = {}", o);
        }
    }

    #[test]
    fn mouth_freezes_when_stopped() {
        let mut mouth = MouthCycle::new();
        mouth.set_moving(true);
        for _ in 0..10 {
            mouth.update(DT);
        }
        let frozen_angle = mouth.angle();
        mouth.set_moving(false);
        for _ in 0..100 {
            mouth.update(DT);
        }
        assert!((mouth.angle() - frozen_angle).abs() < 1e-5);
    }

    #[test]
    fn mouth_custom_frequency() {
        let mut mouth = MouthCycle::new();
        mouth.set_frequency(16.0);
        mouth.set_moving(true);
        // At 16 Hz the mouth completes a full cycle faster.
        mouth.update(DT);
        let fast = mouth.angle();

        let mut slow = MouthCycle::new();
        slow.set_frequency(4.0);
        slow.set_moving(true);
        slow.update(DT);
        // Faster frequency should yield a larger phase advance per tick.
        assert!(fast > slow.angle());
    }

    // ── GhostFloat ───────────────────────────────────────────

    #[test]
    fn float_starts_at_zero() {
        let float = GhostFloat::new();
        assert!(float.offset().abs() < 1e-5);
    }

    #[test]
    fn float_oscillates() {
        let mut float = GhostFloat::new();
        let mut saw_positive = false;
        let mut saw_negative = false;
        for _ in 0..500 {
            float.update(DT);
            if float.offset() > 0.01 {
                saw_positive = true;
            }
            if float.offset() < -0.01 {
                saw_negative = true;
            }
        }
        assert!(saw_positive, "never saw positive offset");
        assert!(saw_negative, "never saw negative offset");
    }

    #[test]
    fn float_within_amplitude() {
        let mut float = GhostFloat::new();
        for _ in 0..1000 {
            float.update(DT);
            assert!(
                float.offset().abs() <= DEFAULT_FLOAT_AMPLITUDE + 1e-5,
                "offset {} exceeds amplitude {}",
                float.offset(),
                DEFAULT_FLOAT_AMPLITUDE,
            );
        }
    }

    #[test]
    fn float_with_phase_offsets_ghosts() {
        let a = GhostFloat::with_phase(0.0);
        let b = GhostFloat::with_phase(PI);
        // At phase=0 offset is 0, at phase=PI offset is ~0 too (sin(PI)≈0),
        // but after one tick they diverge.
        let mut a = a;
        let mut b = b;
        a.update(DT);
        b.update(DT);
        // They should have different offsets.
        assert!(
            (a.offset() - b.offset()).abs() > 1e-5,
            "a={} b={}",
            a.offset(),
            b.offset()
        );
    }

    #[test]
    fn float_phase_wraps() {
        let mut float = GhostFloat::new();
        for _ in 0..10000 {
            float.update(DT);
        }
        assert!(float.phase() < TAU, "phase = {}", float.phase());
    }

    // ── DeathSpiral ──────────────────────────────────────────

    #[test]
    fn death_starts_inactive() {
        let death = DeathSpiral::new();
        assert!(!death.is_active());
        assert!(!death.finished());
    }

    #[test]
    fn death_progress_starts_at_zero() {
        let death = DeathSpiral::new();
        assert!(death.progress().abs() < 1e-5);
    }

    #[test]
    fn death_activates_on_start() {
        let mut death = DeathSpiral::new();
        death.start();
        assert!(death.is_active());
        assert!(!death.finished());
    }

    #[test]
    fn death_progresses_over_time() {
        let mut death = DeathSpiral::new();
        death.start();
        for _ in 0..30 {
            death.update(DT);
        }
        assert!(death.progress() > 0.0);
        assert!(death.progress() < 1.0);
        assert!(death.is_active());
    }

    #[test]
    fn death_finishes_after_duration() {
        let mut death = DeathSpiral::new();
        death.start();
        // Run past the full duration.
        let ticks = (DEFAULT_DEATH_DURATION / DT) as usize + 10;
        for _ in 0..ticks {
            death.update(DT);
        }
        assert!(!death.is_active());
        assert!(death.finished());
        assert!((death.progress() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn death_rotation_increases() {
        let mut death = DeathSpiral::new();
        death.start();
        let mut prev_rotation = 0.0;
        for i in 0..30 {
            death.update(DT);
            if i > 0 {
                assert!(
                    death.rotation() >= prev_rotation,
                    "rotation decreased at tick {}",
                    i
                );
            }
            prev_rotation = death.rotation();
        }
    }

    #[test]
    fn death_collapse_ease_in() {
        let mut death = DeathSpiral::new();
        death.start();
        // At 50% progress, collapse should be 0.25 (t²).
        let half_ticks = ((DEFAULT_DEATH_DURATION / 2.0) / DT) as usize;
        for _ in 0..half_ticks {
            death.update(DT);
        }
        let c = death.collapse();
        // Should be roughly 0.25 (quadratic ease-in at t=0.5).
        assert!(
            (c - 0.25).abs() < 0.05,
            "collapse at ~50% = {}, expected ~0.25",
            c
        );
    }

    #[test]
    fn death_does_not_progress_when_inactive() {
        let mut death = DeathSpiral::new();
        // Don't call start().
        for _ in 0..100 {
            death.update(DT);
        }
        assert!(death.progress().abs() < 1e-5);
        assert!(!death.is_active());
    }

    #[test]
    fn death_reset() {
        let mut death = DeathSpiral::new();
        death.start();
        for _ in 0..30 {
            death.update(DT);
        }
        death.reset();
        assert!(!death.is_active());
        assert!(!death.finished());
        assert!(death.progress().abs() < 1e-5);
    }

    // ── FrightenedFlash ──────────────────────────────────────

    #[test]
    fn flash_not_active_above_threshold() {
        let flash = FrightenedFlash::new();
        assert!(!flash.should_flash(5.0));
    }

    #[test]
    fn flash_active_below_threshold() {
        let flash = FrightenedFlash::new();
        assert!(flash.should_flash(2.0));
    }

    #[test]
    fn flash_not_active_at_zero() {
        let flash = FrightenedFlash::new();
        assert!(!flash.should_flash(0.0));
    }

    #[test]
    fn flash_starts_not_white() {
        let flash = FrightenedFlash::new();
        // Phase is 0 → sin(0) = 0, not > 0.
        assert!(!flash.is_white());
    }

    #[test]
    fn flash_alternates() {
        let mut flash = FrightenedFlash::new();
        let remaining = 2.0; // Below threshold.
        let mut saw_white = false;
        let mut saw_blue = false;
        for _ in 0..500 {
            flash.update(DT, remaining);
            if flash.is_white() {
                saw_white = true;
            } else {
                saw_blue = true;
            }
        }
        assert!(saw_white, "never saw white flash");
        assert!(saw_blue, "never saw blue flash");
    }

    #[test]
    fn flash_does_not_advance_above_threshold() {
        let mut flash = FrightenedFlash::new();
        for _ in 0..100 {
            flash.update(DT, 5.0);
        }
        assert!(!flash.is_white());
    }

    #[test]
    fn flash_resets_phase_above_threshold() {
        let mut flash = FrightenedFlash::new();
        // Advance while flashing.
        for _ in 0..50 {
            flash.update(DT, 2.0);
        }
        // Move above threshold — phase should reset.
        flash.update(DT, 5.0);
        assert!(flash.phase.abs() < 1e-5);
    }

    #[test]
    fn flash_frequency_increases_with_urgency() {
        // Near the threshold (less urgent): slower flashing.
        // Near zero (more urgent): faster flashing.
        let mut slow = FrightenedFlash::new();
        let mut fast = FrightenedFlash::new();

        // Run both for the same number of ticks.
        for _ in 0..100 {
            slow.update(DT, 2.9); // Just entered warn window.
            fast.update(DT, 0.5); // About to expire.
        }

        // The fast one should have accumulated more phase.
        assert!(
            fast.phase > slow.phase,
            "fast.phase={} should exceed slow.phase={}",
            fast.phase,
            slow.phase
        );
    }

    #[test]
    fn flash_reset_clears_phase() {
        let mut flash = FrightenedFlash::new();
        for _ in 0..50 {
            flash.update(DT, 2.0);
        }
        flash.reset();
        assert!(flash.phase.abs() < 1e-5);
        assert!(!flash.is_white());
    }

    #[test]
    fn flash_custom_warn_threshold() {
        let mut flash = FrightenedFlash::new();
        flash.set_warn_threshold(5.0);
        assert_eq!(flash.warn_threshold(), 5.0);
        assert!(flash.should_flash(4.0));
    }
}
