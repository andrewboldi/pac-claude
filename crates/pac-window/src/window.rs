use std::sync::Arc;

use crate::input::InputState;
use pac_render::{wgpu, DepthBuffer, GpuContext, TrianglePipeline};
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

struct RenderState {
    gpu: GpuContext<'static>,
    depth: DepthBuffer,
    triangle: TrianglePipeline,
}

struct App {
    config: WindowConfig,
    window: Option<Arc<Window>>,
    render: Option<RenderState>,
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
                    let window = Arc::new(window);
                    let size = window.inner_size();
                    log::info!("Window created: {}x{}", size.width, size.height);

                    let gpu = pollster::block_on(GpuContext::new(
                        window.clone(),
                        size.width,
                        size.height,
                    ));
                    let depth = DepthBuffer::new(&gpu.device, size.width, size.height);
                    let triangle = TrianglePipeline::new(&gpu.device, gpu.format());
                    self.render = Some(RenderState {
                        gpu,
                        depth,
                        triangle,
                    });

                    window.request_redraw();
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
                if let Some(render) = &mut self.render {
                    render.gpu.resize(width, height);
                    if width > 0 && height > 0 {
                        render.depth.resize(&render.gpu.device, width, height);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.input.begin_frame();

                if let Some(render) = &mut self.render {
                    match render.triangle.render_frame(&render.gpu, &render.depth) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            let (w, h) = render.gpu.size();
                            render.gpu.resize(w, h);
                            render.depth.resize(&render.gpu.device, w, h);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of GPU memory");
                            event_loop.exit();
                        }
                        Err(e) => {
                            log::warn!("Surface error: {e:?}");
                        }
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_config_default_values() {
        let config = WindowConfig::default();
        assert_eq!(config.title, "Pac-Man 3D");
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }

    #[test]
    fn window_config_custom_values() {
        let config = WindowConfig {
            title: "Test Window".to_string(),
            width: 1920,
            height: 1080,
        };
        assert_eq!(config.title, "Test Window");
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn window_config_zero_dimensions() {
        let config = WindowConfig {
            title: String::new(),
            width: 0,
            height: 0,
        };
        assert_eq!(config.width, 0);
        assert_eq!(config.height, 0);
        assert!(config.title.is_empty());
    }
}

/// Run the window event loop with the given configuration.
pub fn run(config: WindowConfig) {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App {
        config,
        window: None,
        render: None,
        input: InputState::new(),
    };
    event_loop.run_app(&mut app).expect("event loop error");
}
