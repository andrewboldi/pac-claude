use std::collections::VecDeque;

/// Sound events that can be triggered during gameplay.
///
/// Each variant maps to a distinct audio cue. Events are enqueued
/// by game-logic code and drained by the audio subsystem each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundEvent {
    PelletEat,
    PowerPelletEat,
    GhostEaten,
    PacManDeath,
    LevelStart,
    LevelComplete,
    ExtraLife,
}

/// A simple FIFO queue for sound events.
///
/// Game logic pushes events as they occur; the audio backend
/// drains them once per frame via [`EventQueue::drain`].
#[derive(Debug, Default)]
pub struct EventQueue {
    events: VecDeque<SoundEvent>,
}

impl EventQueue {
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
        }
    }

    /// Push a sound event onto the queue.
    pub fn push(&mut self, event: SoundEvent) {
        self.events.push_back(event);
    }

    /// Drain all queued events, returning an iterator.
    pub fn drain(&mut self) -> impl Iterator<Item = SoundEvent> + '_ {
        self.events.drain(..)
    }

    /// Returns `true` if no events are queued.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the number of queued events.
    pub fn len(&self) -> usize {
        self.events.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_drain() {
        let mut queue = EventQueue::new();
        queue.push(SoundEvent::PelletEat);
        queue.push(SoundEvent::GhostEaten);

        let events: Vec<_> = queue.drain().collect();
        assert_eq!(events, vec![SoundEvent::PelletEat, SoundEvent::GhostEaten]);
        assert!(queue.is_empty());
    }

    #[test]
    fn default_is_empty() {
        let queue = EventQueue::default();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn drain_empties_queue() {
        let mut queue = EventQueue::new();
        queue.push(SoundEvent::LevelStart);
        queue.push(SoundEvent::PelletEat);
        queue.push(SoundEvent::LevelComplete);

        assert_eq!(queue.len(), 3);
        let _ = queue.drain().count();
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn fifo_ordering() {
        let mut queue = EventQueue::new();
        queue.push(SoundEvent::LevelStart);
        queue.push(SoundEvent::PelletEat);
        queue.push(SoundEvent::PowerPelletEat);
        queue.push(SoundEvent::ExtraLife);

        let events: Vec<_> = queue.drain().collect();
        assert_eq!(
            events,
            vec![
                SoundEvent::LevelStart,
                SoundEvent::PelletEat,
                SoundEvent::PowerPelletEat,
                SoundEvent::ExtraLife,
            ]
        );
    }
}
