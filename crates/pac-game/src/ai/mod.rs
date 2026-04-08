//! Ghost AI modules — one sub-module per ghost personality.
//!
//! Each ghost has a dedicated AI module that calls [`Ghost::set_direction`]
//! each tick. The ghost entity handles movement mechanics; the AI decides
//! *where* to go.

pub mod blinky;
pub mod pinky;

pub use blinky::BlinkyAi;
