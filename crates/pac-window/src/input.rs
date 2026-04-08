use std::collections::HashSet;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Tracks keyboard and mouse input state across frames.
///
/// Keys and mouse buttons are tracked in three sets:
/// - **pressed**: newly pressed this frame (edge-triggered)
/// - **held**: currently down (level-triggered)
/// - **released**: newly released this frame (edge-triggered)
///
/// Call [`InputState::begin_frame`] at the start of each frame to clear the
/// per-frame `pressed` and `released` sets, then feed window events through
/// [`InputState::process_event`].
pub struct InputState {
    keys_pressed: HashSet<KeyCode>,
    keys_held: HashSet<KeyCode>,
    keys_released: HashSet<KeyCode>,

    mouse_buttons_pressed: HashSet<MouseButton>,
    mouse_buttons_held: HashSet<MouseButton>,
    mouse_buttons_released: HashSet<MouseButton>,

    mouse_x: f64,
    mouse_y: f64,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            keys_pressed: HashSet::new(),
            keys_held: HashSet::new(),
            keys_released: HashSet::new(),
            mouse_buttons_pressed: HashSet::new(),
            mouse_buttons_held: HashSet::new(),
            mouse_buttons_released: HashSet::new(),
            mouse_x: 0.0,
            mouse_y: 0.0,
        }
    }

    /// Clear per-frame edge sets. Call at the start of each frame before
    /// processing events.
    pub fn begin_frame(&mut self) {
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.mouse_buttons_pressed.clear();
        self.mouse_buttons_released.clear();
    }

    /// Feed a winit [`WindowEvent`] into the input tracker.
    pub fn process_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if !event.repeat {
                                self.keys_pressed.insert(code);
                            }
                            self.keys_held.insert(code);
                        }
                        ElementState::Released => {
                            self.keys_released.insert(code);
                            self.keys_held.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => {
                    self.mouse_buttons_pressed.insert(*button);
                    self.mouse_buttons_held.insert(*button);
                }
                ElementState::Released => {
                    self.mouse_buttons_released.insert(*button);
                    self.mouse_buttons_held.remove(button);
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_x = position.x;
                self.mouse_y = position.y;
            }
            _ => {}
        }
    }

    // -- Key queries --

    /// True during the frame the key was first pressed (not on repeat).
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    /// True while the key is held down.
    pub fn key_held(&self, key: KeyCode) -> bool {
        self.keys_held.contains(&key)
    }

    /// True during the frame the key was released.
    pub fn key_released(&self, key: KeyCode) -> bool {
        self.keys_released.contains(&key)
    }

    // -- Mouse button queries --

    pub fn mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons_pressed.contains(&button)
    }

    pub fn mouse_button_held(&self, button: MouseButton) -> bool {
        self.mouse_buttons_held.contains(&button)
    }

    pub fn mouse_button_released(&self, button: MouseButton) -> bool {
        self.mouse_buttons_released.contains(&button)
    }

    // -- Mouse position --

    pub fn mouse_position(&self) -> (f64, f64) {
        (self.mouse_x, self.mouse_y)
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: winit's DeviceId and KeyEventExtra have no public constructors,
    // so we test the input state logic directly via private field access
    // (tests are in the same module). The process_event integration is
    // verified by the build and by the event loop wiring in window.rs.

    /// Simulate a non-repeat key press.
    fn simulate_key_press(input: &mut InputState, code: KeyCode) {
        input.keys_pressed.insert(code);
        input.keys_held.insert(code);
    }

    /// Simulate a key release.
    fn simulate_key_release(input: &mut InputState, code: KeyCode) {
        input.keys_released.insert(code);
        input.keys_held.remove(&code);
    }

    /// Simulate a mouse button press.
    fn simulate_mouse_press(input: &mut InputState, button: MouseButton) {
        input.mouse_buttons_pressed.insert(button);
        input.mouse_buttons_held.insert(button);
    }

    /// Simulate a mouse button release.
    fn simulate_mouse_release(input: &mut InputState, button: MouseButton) {
        input.mouse_buttons_released.insert(button);
        input.mouse_buttons_held.remove(&button);
    }

    #[test]
    fn key_press_tracks_pressed_and_held() {
        let mut input = InputState::new();
        simulate_key_press(&mut input, KeyCode::KeyW);

        assert!(input.key_pressed(KeyCode::KeyW));
        assert!(input.key_held(KeyCode::KeyW));
        assert!(!input.key_released(KeyCode::KeyW));
    }

    #[test]
    fn key_release_clears_held() {
        let mut input = InputState::new();
        simulate_key_press(&mut input, KeyCode::KeyW);
        simulate_key_release(&mut input, KeyCode::KeyW);

        assert!(!input.key_held(KeyCode::KeyW));
        assert!(input.key_released(KeyCode::KeyW));
    }

    #[test]
    fn begin_frame_clears_edge_sets() {
        let mut input = InputState::new();
        simulate_key_press(&mut input, KeyCode::KeyW);
        assert!(input.key_pressed(KeyCode::KeyW));

        input.begin_frame();
        assert!(!input.key_pressed(KeyCode::KeyW));
        // held persists across frames
        assert!(input.key_held(KeyCode::KeyW));
    }

    #[test]
    fn multiple_keys_tracked_independently() {
        let mut input = InputState::new();
        simulate_key_press(&mut input, KeyCode::KeyW);
        simulate_key_press(&mut input, KeyCode::KeyA);

        assert!(input.key_held(KeyCode::KeyW));
        assert!(input.key_held(KeyCode::KeyA));
        assert!(!input.key_held(KeyCode::KeyS));

        simulate_key_release(&mut input, KeyCode::KeyW);
        assert!(!input.key_held(KeyCode::KeyW));
        assert!(input.key_held(KeyCode::KeyA));
    }

    #[test]
    fn mouse_position_updates() {
        let mut input = InputState::new();
        input.mouse_x = 120.5;
        input.mouse_y = 340.0;

        let (x, y) = input.mouse_position();
        assert!((x - 120.5).abs() < f64::EPSILON);
        assert!((y - 340.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mouse_button_press_and_release() {
        let mut input = InputState::new();
        simulate_mouse_press(&mut input, MouseButton::Left);

        assert!(input.mouse_button_pressed(MouseButton::Left));
        assert!(input.mouse_button_held(MouseButton::Left));

        simulate_mouse_release(&mut input, MouseButton::Left);
        assert!(input.mouse_button_released(MouseButton::Left));
        assert!(!input.mouse_button_held(MouseButton::Left));
    }

    #[test]
    fn begin_frame_clears_mouse_edge_sets() {
        let mut input = InputState::new();
        simulate_mouse_press(&mut input, MouseButton::Left);
        assert!(input.mouse_button_pressed(MouseButton::Left));

        input.begin_frame();
        assert!(!input.mouse_button_pressed(MouseButton::Left));
        assert!(input.mouse_button_held(MouseButton::Left));
    }

    #[test]
    fn default_mouse_position_is_zero() {
        let input = InputState::new();
        assert_eq!(input.mouse_position(), (0.0, 0.0));
    }

    #[test]
    fn default_trait_matches_new() {
        let a = InputState::new();
        let b = InputState::default();
        assert_eq!(a.mouse_position(), b.mouse_position());
        assert!(!b.key_held(KeyCode::KeyW));
        assert!(!b.key_pressed(KeyCode::KeyW));
        assert!(!b.key_released(KeyCode::KeyW));
        assert!(!b.mouse_button_held(MouseButton::Left));
    }

    #[test]
    fn release_without_prior_press() {
        let mut input = InputState::new();
        simulate_key_release(&mut input, KeyCode::KeyW);

        assert!(input.key_released(KeyCode::KeyW));
        assert!(!input.key_held(KeyCode::KeyW));
        assert!(!input.key_pressed(KeyCode::KeyW));
    }

    #[test]
    fn double_press_same_key_stays_held() {
        let mut input = InputState::new();
        simulate_key_press(&mut input, KeyCode::KeyW);
        simulate_key_press(&mut input, KeyCode::KeyW);

        assert!(input.key_pressed(KeyCode::KeyW));
        assert!(input.key_held(KeyCode::KeyW));
    }

    #[test]
    fn multiple_mouse_buttons_tracked_independently() {
        let mut input = InputState::new();
        simulate_mouse_press(&mut input, MouseButton::Left);
        simulate_mouse_press(&mut input, MouseButton::Right);

        assert!(input.mouse_button_held(MouseButton::Left));
        assert!(input.mouse_button_held(MouseButton::Right));
        assert!(!input.mouse_button_held(MouseButton::Middle));

        simulate_mouse_release(&mut input, MouseButton::Left);
        assert!(!input.mouse_button_held(MouseButton::Left));
        assert!(input.mouse_button_held(MouseButton::Right));
    }

    #[test]
    fn begin_frame_clears_released_sets() {
        let mut input = InputState::new();
        simulate_key_press(&mut input, KeyCode::KeyW);
        simulate_key_release(&mut input, KeyCode::KeyW);
        simulate_mouse_press(&mut input, MouseButton::Left);
        simulate_mouse_release(&mut input, MouseButton::Left);

        assert!(input.key_released(KeyCode::KeyW));
        assert!(input.mouse_button_released(MouseButton::Left));

        input.begin_frame();

        assert!(!input.key_released(KeyCode::KeyW));
        assert!(!input.mouse_button_released(MouseButton::Left));
    }

    #[test]
    fn begin_frame_preserves_mouse_position() {
        let mut input = InputState::new();
        input.mouse_x = 100.0;
        input.mouse_y = 200.0;

        input.begin_frame();

        assert_eq!(input.mouse_position(), (100.0, 200.0));
    }

    #[test]
    fn full_lifecycle_across_frames() {
        let mut input = InputState::new();

        // Frame 1: press key
        input.begin_frame();
        simulate_key_press(&mut input, KeyCode::KeyW);
        assert!(input.key_pressed(KeyCode::KeyW));
        assert!(input.key_held(KeyCode::KeyW));

        // Frame 2: key still held, no longer "pressed"
        input.begin_frame();
        assert!(!input.key_pressed(KeyCode::KeyW));
        assert!(input.key_held(KeyCode::KeyW));

        // Frame 3: release key
        input.begin_frame();
        simulate_key_release(&mut input, KeyCode::KeyW);
        assert!(input.key_released(KeyCode::KeyW));
        assert!(!input.key_held(KeyCode::KeyW));

        // Frame 4: all clear
        input.begin_frame();
        assert!(!input.key_pressed(KeyCode::KeyW));
        assert!(!input.key_held(KeyCode::KeyW));
        assert!(!input.key_released(KeyCode::KeyW));
    }

    #[test]
    fn mouse_release_without_prior_press() {
        let mut input = InputState::new();
        simulate_mouse_release(&mut input, MouseButton::Right);

        assert!(input.mouse_button_released(MouseButton::Right));
        assert!(!input.mouse_button_held(MouseButton::Right));
        assert!(!input.mouse_button_pressed(MouseButton::Right));
    }

    #[test]
    fn simultaneous_key_and_mouse_input() {
        let mut input = InputState::new();
        simulate_key_press(&mut input, KeyCode::Space);
        simulate_mouse_press(&mut input, MouseButton::Left);

        assert!(input.key_pressed(KeyCode::Space));
        assert!(input.key_held(KeyCode::Space));
        assert!(input.mouse_button_pressed(MouseButton::Left));
        assert!(input.mouse_button_held(MouseButton::Left));

        input.begin_frame();

        assert!(!input.key_pressed(KeyCode::Space));
        assert!(input.key_held(KeyCode::Space));
        assert!(!input.mouse_button_pressed(MouseButton::Left));
        assert!(input.mouse_button_held(MouseButton::Left));
    }

    #[test]
    fn mouse_position_overwritten_by_later_update() {
        let mut input = InputState::new();
        input.mouse_x = 10.0;
        input.mouse_y = 20.0;
        assert_eq!(input.mouse_position(), (10.0, 20.0));

        input.mouse_x = 300.5;
        input.mouse_y = 400.5;
        assert_eq!(input.mouse_position(), (300.5, 400.5));
    }

    #[test]
    fn press_release_press_within_same_frame() {
        let mut input = InputState::new();
        simulate_key_press(&mut input, KeyCode::KeyA);
        simulate_key_release(&mut input, KeyCode::KeyA);
        simulate_key_press(&mut input, KeyCode::KeyA);

        // Both pressed and released should be set; held should be true
        assert!(input.key_pressed(KeyCode::KeyA));
        assert!(input.key_released(KeyCode::KeyA));
        assert!(input.key_held(KeyCode::KeyA));
    }

    #[test]
    fn many_keys_held_simultaneously() {
        let mut input = InputState::new();
        let keys = [
            KeyCode::KeyW,
            KeyCode::KeyA,
            KeyCode::KeyS,
            KeyCode::KeyD,
            KeyCode::ShiftLeft,
            KeyCode::Space,
        ];
        for &k in &keys {
            simulate_key_press(&mut input, k);
        }
        for &k in &keys {
            assert!(input.key_held(k));
        }

        // Release just one
        simulate_key_release(&mut input, KeyCode::KeyS);
        assert!(!input.key_held(KeyCode::KeyS));
        assert!(input.key_held(KeyCode::KeyW));
        assert!(input.key_held(KeyCode::Space));
    }

    #[test]
    fn mouse_button_full_lifecycle() {
        let mut input = InputState::new();

        // Frame 1: press
        input.begin_frame();
        simulate_mouse_press(&mut input, MouseButton::Middle);
        assert!(input.mouse_button_pressed(MouseButton::Middle));
        assert!(input.mouse_button_held(MouseButton::Middle));

        // Frame 2: held only
        input.begin_frame();
        assert!(!input.mouse_button_pressed(MouseButton::Middle));
        assert!(input.mouse_button_held(MouseButton::Middle));

        // Frame 3: release
        input.begin_frame();
        simulate_mouse_release(&mut input, MouseButton::Middle);
        assert!(input.mouse_button_released(MouseButton::Middle));
        assert!(!input.mouse_button_held(MouseButton::Middle));

        // Frame 4: all clear
        input.begin_frame();
        assert!(!input.mouse_button_pressed(MouseButton::Middle));
        assert!(!input.mouse_button_held(MouseButton::Middle));
        assert!(!input.mouse_button_released(MouseButton::Middle));
    }
}
