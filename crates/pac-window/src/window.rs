use crate::input::InputState;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

/// Configuration for window creation.
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Pac-Man 3D".to_string(),
            width: 800,
            height: 600,
        }
    }
}

struct App {
    config: WindowConfig,
    window: Option<Window>,
    input: InputState,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = WindowAttributes::default()
                .with_title(&self.config.title)
                .with_inner_size(LogicalSize::new(self.config.width, self.config.height));
            match event_loop.create_window(attrs) {
                Ok(window) => {
                    log::info!("Window created: {}x{}", self.config.width, self.config.height);
                    self.window = Some(window);
                }
                Err(e) => {
                    log::error!("Failed to create window: {e}");
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        self.input.process_event(&event);

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, exiting");
                event_loop.exit();
            }
            WindowEvent::Resized(PhysicalSize { width, height }) => {
                log::info!("Window resized to {width}x{height}");
            }
            WindowEvent::RedrawRequested => {
                self.input.begin_frame();
            }
            _ => {}
        }
    }
}

/// Run the window event loop with the given configuration.
pub fn run(config: WindowConfig) {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App {
        config,
        window: None,
        input: InputState::new(),
    };
    event_loop.run_app(&mut app).expect("event loop error");
}
