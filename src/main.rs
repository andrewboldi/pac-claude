//! Pac-Man 3D — final game integration.
//!
//! Wires together window, GPU, renderer, maze, entities, collision, ghost AI,
//! game state, animations, UI overlay, and sound events into a single main loop
//! with fixed-timestep physics.

use std::sync::Arc;

use pac_math::glam::Vec3;
use pac_game::ai::BlinkyAi;
use pac_game::animation::{DeathSpiral, FrightenedFlash, GhostFloat, MouthCycle};
use pac_game::collision::{self, CollisionEvent, GhostCollider};
use pac_game::ghost::{Ghost, GhostId, GhostState};
use pac_game::ghost_mode::GhostModeController;
use pac_game::maze::MazeData;
use pac_game::maze_renderer::{self, MazeMeshConfig, MazeScene};
use pac_game::pacman::{Direction, PacMan};
use pac_game::pellet::PelletManager;
use pac_game::sound::{EventQueue, SoundEvent};
use pac_game::ui::UiOverlay;
use pac_math::Transform;
use pac_render::{
    Camera, DepthBuffer, GpuContext, LightManager, Material, Mesh, Renderer, SceneGraph,
};
use pac_window::time::TimeState;
use pac_window::InputState;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::{Window, WindowAttributes, WindowId};

// ── Game state machine ──────────────────────────────────────────────

/// Top-level game phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GamePhase {
    /// Showing "READY!" before play begins.
    Ready,
    /// Normal gameplay.
    Playing,
    /// Pac-Man death animation playing.
    Dying,
    /// "GAME OVER" screen.
    GameOver,
}

/// Duration of the "READY!" splash in seconds.
const READY_DURATION: f32 = 2.0;
/// Duration of the "GAME OVER" splash before accepting restart.
const GAME_OVER_DURATION: f32 = 3.0;
/// Starting lives.
const STARTING_LIVES: u32 = 3;
/// Score threshold for an extra life.
const EXTRA_LIFE_THRESHOLD: u32 = 10_000;
/// Points for collecting a normal pellet.
const PELLET_SCORE: u32 = 10;
/// Points for collecting a power pellet.
const POWER_PELLET_SCORE: u32 = 50;

// ── Ghost spawn positions (classic Pac-Man layout) ──────────────────

const BLINKY_SPAWN: (usize, usize) = (13, 11);
const PINKY_SPAWN: (usize, usize) = (13, 14);
const INKY_SPAWN: (usize, usize) = (11, 14);
const CLYDE_SPAWN: (usize, usize) = (15, 14);

// ── Camera setup ────────────────────────────────────────────────────

fn default_camera() -> Camera {
    let mut cam = Camera::default();
    cam.position = Vec3::new(13.5, 35.0, 20.0);
    cam.pitch = -1.1;
    cam.yaw = std::f32::consts::FRAC_PI_2;
    cam.fov_y = std::f32::consts::FRAC_PI_4;
    cam.near = 0.1;
    cam.far = 200.0;
    cam
}

// ── GPU render state ────────────────────────────────────────────────

struct RenderState {
    gpu: GpuContext<'static>,
    depth: DepthBuffer,
    renderer: Renderer,
    lights: LightManager,
    scene: SceneGraph,
    camera: Camera,
    materials: Vec<Material>,
    meshes: Vec<pac_render::GpuMesh>,
    default_material: Material,
    _maze_scene: MazeScene,
    ui: UiOverlay,
    pacman_node: pac_render::NodeHandle,
    ghost_nodes: [pac_render::NodeHandle; 4],
}

// ── Game state ──────────────────────────────────────────────────────

struct GameState {
    maze: MazeData,
    pacman: PacMan,
    ghosts: Vec<Ghost>,
    pellets: PelletManager,
    mode_controller: GhostModeController,
    blinky_ai: BlinkyAi,
    mouth: MouthCycle,
    ghost_floats: [GhostFloat; 4],
    death_spiral: DeathSpiral,
    frightened_flash: FrightenedFlash,
    sound_queue: EventQueue,
    phase: GamePhase,
    phase_timer: f32,
    score: u32,
    lives: u32,
    extra_life_awarded: bool,
}

impl GameState {
    fn new(maze: MazeData) -> Self {
        let pacman = PacMan::from_maze(&maze).expect("maze must have a PlayerSpawn tile");
        let ghosts = vec![
            Ghost::with_id(GhostId::Blinky, BLINKY_SPAWN.0, BLINKY_SPAWN.1),
            Ghost::new_in_house(GhostId::Pinky, PINKY_SPAWN.0, PINKY_SPAWN.1),
            Ghost::new_in_house(GhostId::Inky, INKY_SPAWN.0, INKY_SPAWN.1),
            Ghost::new_in_house(GhostId::Clyde, CLYDE_SPAWN.0, CLYDE_SPAWN.1),
        ];
        let pellets = PelletManager::from_maze(&maze);

        Self {
            maze,
            pacman,
            ghosts,
            pellets,
            mode_controller: GhostModeController::new(),
            blinky_ai: BlinkyAi::new(),
            mouth: MouthCycle::new(),
            ghost_floats: [
                GhostFloat::new(),
                GhostFloat::with_phase(std::f32::consts::FRAC_PI_2),
                GhostFloat::with_phase(std::f32::consts::PI),
                GhostFloat::with_phase(3.0 * std::f32::consts::FRAC_PI_2),
            ],
            death_spiral: DeathSpiral::new(),
            frightened_flash: FrightenedFlash::new(),
            sound_queue: EventQueue::new(),
            phase: GamePhase::Ready,
            phase_timer: 0.0,
            score: 0,
            lives: STARTING_LIVES,
            extra_life_awarded: false,
        }
    }

    fn reset_positions(&mut self) {
        self.pacman = PacMan::from_maze(&self.maze).expect("maze must have a PlayerSpawn");
        self.ghosts = vec![
            Ghost::with_id(GhostId::Blinky, BLINKY_SPAWN.0, BLINKY_SPAWN.1),
            Ghost::new_in_house(GhostId::Pinky, PINKY_SPAWN.0, PINKY_SPAWN.1),
            Ghost::new_in_house(GhostId::Inky, INKY_SPAWN.0, INKY_SPAWN.1),
            Ghost::new_in_house(GhostId::Clyde, CLYDE_SPAWN.0, CLYDE_SPAWN.1),
        ];
        self.mode_controller.reset();
        self.blinky_ai = BlinkyAi::new();
        self.mouth = MouthCycle::new();
        self.death_spiral.reset();
        self.frightened_flash.reset();
        self.phase = GamePhase::Ready;
        self.phase_timer = 0.0;
    }

    fn reset_game(&mut self) {
        self.pellets.reset();
        self.score = 0;
        self.lives = STARTING_LIVES;
        self.extra_life_awarded = false;
        self.reset_positions();
    }
}

// ── Free functions for game logic (avoids borrow conflicts) ─────────

/// Process keyboard input and map to game actions.
fn process_input(input: &InputState, game: &mut GameState) {
    if game.phase != GamePhase::Playing {
        return;
    }
    if input.key_held(KeyCode::ArrowUp) || input.key_held(KeyCode::KeyW) {
        game.pacman.set_direction(Direction::Up);
    }
    if input.key_held(KeyCode::ArrowDown) || input.key_held(KeyCode::KeyS) {
        game.pacman.set_direction(Direction::Down);
    }
    if input.key_held(KeyCode::ArrowLeft) || input.key_held(KeyCode::KeyA) {
        game.pacman.set_direction(Direction::Left);
    }
    if input.key_held(KeyCode::ArrowRight) || input.key_held(KeyCode::KeyD) {
        game.pacman.set_direction(Direction::Right);
    }
}

/// Run one fixed-timestep physics update.
fn fixed_update(game: &mut GameState, dt: f32) {
    match game.phase {
        GamePhase::Ready => {
            game.phase_timer += dt;
            if game.phase_timer >= READY_DURATION {
                game.phase = GamePhase::Playing;
                game.phase_timer = 0.0;
                game.sound_queue.push(SoundEvent::LevelStart);
            }
        }
        GamePhase::Playing => {
            update_playing(game, dt);
        }
        GamePhase::Dying => {
            game.death_spiral.update(dt);
            if game.death_spiral.finished() {
                if game.lives == 0 {
                    game.phase = GamePhase::GameOver;
                    game.phase_timer = 0.0;
                } else {
                    game.reset_positions();
                }
            }
        }
        GamePhase::GameOver => {
            game.phase_timer += dt;
        }
    }
}

/// Update during active gameplay.
fn update_playing(game: &mut GameState, dt: f32) {
    // Update Pac-Man movement.
    game.pacman.update(dt, &game.maze);

    // Update mouth animation.
    game.mouth.set_moving(game.pacman.current_dir().is_some());
    game.mouth.update(dt);

    // Update ghost mode controller (scatter/chase/frightened timers).
    game.mode_controller.update(dt, &mut game.ghosts);

    // Update ghost AI and movement.
    let remaining = game.pellets.remaining_count();
    game.blinky_ai
        .update(&mut game.ghosts[0], &game.pacman, &game.maze, remaining);
    pac_game::ai::pinky::update(&mut game.ghosts[1], &game.pacman, &game.maze);
    pac_game::ai::pinky::update(&mut game.ghosts[2], &game.pacman, &game.maze);
    pac_game::ai::pinky::update(&mut game.ghosts[3], &game.pacman, &game.maze);

    for ghost in game.ghosts.iter_mut() {
        ghost.update(dt, &game.maze);
    }

    // Update ghost float animations.
    for float in game.ghost_floats.iter_mut() {
        float.update(dt);
    }

    // Update frightened flash.
    let fright_remaining = game.mode_controller.frightened_time_remaining();
    game.frightened_flash.update(dt, fright_remaining);

    // Build ghost colliders for collision check.
    let ghost_colliders: Vec<GhostCollider> = game
        .ghosts
        .iter()
        .map(|g| GhostCollider {
            grid_x: g.grid_x(),
            grid_y: g.grid_y(),
            frightened: g.mode() == GhostState::Frightened,
        })
        .collect();

    // Check collisions.
    let events = collision::check_collisions(&game.pacman, &mut game.pellets, &ghost_colliders);

    for event in &events {
        match event {
            CollisionEvent::PelletCollected { .. } => {
                game.score += PELLET_SCORE;
                game.sound_queue.push(SoundEvent::PelletEat);
            }
            CollisionEvent::PowerPelletCollected { .. } => {
                game.score += POWER_PELLET_SCORE;
                game.sound_queue.push(SoundEvent::PowerPelletEat);
                game.mode_controller.trigger_frightened(&mut game.ghosts);
                game.frightened_flash.reset();
            }
            CollisionEvent::GhostHit { .. } => {
                game.sound_queue.push(SoundEvent::PacManDeath);
                game.lives = game.lives.saturating_sub(1);
                game.phase = GamePhase::Dying;
                game.death_spiral.start();
                return;
            }
            CollisionEvent::GhostEaten { ghost_index } => {
                game.mode_controller.notify_ghost_eaten();
                let eat_score = game.mode_controller.last_eat_score();
                game.score += eat_score;
                game.ghosts[*ghost_index].set_mode(GhostState::Eaten);
                game.sound_queue.push(SoundEvent::GhostEaten);
            }
        }
    }

    // Check for extra life.
    if !game.extra_life_awarded && game.score >= EXTRA_LIFE_THRESHOLD {
        game.extra_life_awarded = true;
        game.lives += 1;
        game.sound_queue.push(SoundEvent::ExtraLife);
    }

    // Check for level complete.
    if game.pellets.all_collected() {
        game.sound_queue.push(SoundEvent::LevelComplete);
        game.pellets.reset();
        game.reset_positions();
    }

    // Handle eaten ghosts returning to ghost house.
    for i in 0..game.ghosts.len() {
        if game.ghosts[i].mode() == GhostState::Eaten
            && game.ghosts[i].grid_x() == game.ghosts[i].spawn_x()
            && game.ghosts[i].grid_y() == game.ghosts[i].spawn_y()
            && game.ghosts[i].move_progress() >= 1.0
        {
            game.mode_controller
                .return_ghost_to_base(&mut game.ghosts[i]);
        }
    }

    // Drain sound events (no audio backend yet).
    for _event in game.sound_queue.drain() {
        // Future: play actual sounds.
    }
}

/// Update scene graph transforms from game state for rendering.
fn sync_scene(render: &mut RenderState, game: &GameState, alpha: f64) {
    let alpha = alpha as f32;

    // Update Pac-Man scene node position.
    let (pac_x, pac_z) = game.pacman.world_position(alpha);
    render
        .scene
        .node_mut(render.pacman_node)
        .transform
        .position = Vec3::new(pac_x, 0.0, pac_z);

    // Update ghost scene node positions.
    for (i, ghost) in game.ghosts.iter().enumerate() {
        let (gx, gz) = ghost.world_position(alpha);
        let float_y = game.ghost_floats[i].offset();
        render
            .scene
            .node_mut(render.ghost_nodes[i])
            .transform
            .position = Vec3::new(gx, float_y, gz);
    }

    // Update camera aspect ratio.
    let (w, h) = render.gpu.size();
    if w > 0 && h > 0 {
        render.camera.aspect = w as f32 / h as f32;
    }

    // Propagate world matrices.
    render.scene.update_world_matrices();

    // Upload light state.
    render.lights.write(&render.gpu.queue);
}

/// Render the frame: 3D scene + UI overlay.
fn render_frame(
    render: &mut RenderState,
    game: &GameState,
) -> Result<(), wgpu::SurfaceError> {
    // Render the 3D scene.
    render.renderer.render_frame(
        &render.gpu,
        &render.depth,
        &render.camera,
        &render.lights,
        &render.scene,
        &render.meshes,
        &render.materials,
        &render.default_material,
    )?;

    // --- UI overlay ---
    render.ui.begin_frame();
    render.ui.draw_score(game.score);
    render.ui.draw_lives(game.lives);

    match game.phase {
        GamePhase::Ready => render.ui.draw_ready(),
        GamePhase::GameOver => render.ui.draw_game_over(),
        _ => {}
    }

    // UI vertex upload and render pass would go here once the UI pipeline
    // is integrated into the Renderer's command encoder. The draw_* calls
    // above populate the vertex buffers; a future pass will present them.

    Ok(())
}

// ── Application ─────────────────────────────────────────────────────

struct App {
    window: Option<Arc<Window>>,
    render: Option<RenderState>,
    game: Option<GameState>,
    input: InputState,
    time: TimeState,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            render: None,
            game: None,
            input: InputState::new(),
            time: TimeState::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Pac-Man 3D")
            .with_inner_size(LogicalSize::new(1024u32, 768u32));

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("Failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        let size = window.inner_size();
        log::info!("Window created: {}x{}", size.width, size.height);

        // ── Initialize GPU ──────────────────────────────────────
        let gpu = pollster::block_on(GpuContext::new(
            window.clone(),
            size.width,
            size.height,
        ));
        let depth = DepthBuffer::new(&gpu.device, size.width, size.height);

        // ── Load maze ───────────────────────────────────────────
        let maze_json = include_bytes!("../assets/maze/classic.json");
        let maze = MazeData::from_json(maze_json).expect("failed to parse classic maze");

        // ── Create renderer ─────────────────────────────────────
        let renderer = Renderer::new(&gpu.device, gpu.format());
        let mat_layout = renderer.material_layout();

        // ── Create lights ───────────────────────────────────────
        let mut lights = LightManager::new(&gpu.device);
        lights.ambient = Vec3::splat(0.2);
        lights.directional.direction = Vec3::new(0.3, 1.0, 0.5).normalize();
        lights.directional.color = Vec3::ONE;
        lights.directional.intensity = 0.8;

        // ── Create meshes ───────────────────────────────────────
        let cube_mesh = Mesh::cube().upload(&gpu.device, "cube");
        let plane_mesh = Mesh::plane(1).upload(&gpu.device, "plane");
        let sphere_mesh = Mesh::sphere(16, 12).upload(&gpu.device, "entity_sphere");
        let meshes = vec![cube_mesh, plane_mesh, sphere_mesh];

        // ── Create materials ────────────────────────────────────
        // 0=wall, 1=floor, 2=ghost_house, 3=ghost_door,
        // 4=pacman, 5=blinky, 6=pinky, 7=inky, 8=clyde
        let materials = vec![
            Material::from_color(&gpu.device, &gpu.queue, mat_layout,
                [0.1, 0.1, 0.5], [0.3, 0.3, 0.6], 64.0),
            Material::from_color(&gpu.device, &gpu.queue, mat_layout,
                [0.05, 0.05, 0.1], [0.1, 0.1, 0.1], 16.0),
            Material::from_color(&gpu.device, &gpu.queue, mat_layout,
                [0.2, 0.1, 0.3], [0.2, 0.2, 0.2], 32.0),
            Material::from_color(&gpu.device, &gpu.queue, mat_layout,
                [0.8, 0.4, 0.6], [0.3, 0.3, 0.3], 32.0),
            Material::from_color(&gpu.device, &gpu.queue, mat_layout,
                [1.0, 0.9, 0.0], [0.5, 0.5, 0.3], 32.0),
            Material::from_color(&gpu.device, &gpu.queue, mat_layout,
                [1.0, 0.0, 0.0], [0.3, 0.3, 0.3], 32.0),
            Material::from_color(&gpu.device, &gpu.queue, mat_layout,
                [1.0, 0.7, 0.8], [0.3, 0.3, 0.3], 32.0),
            Material::from_color(&gpu.device, &gpu.queue, mat_layout,
                [0.0, 1.0, 1.0], [0.3, 0.3, 0.3], 32.0),
            Material::from_color(&gpu.device, &gpu.queue, mat_layout,
                [1.0, 0.6, 0.0], [0.3, 0.3, 0.3], 32.0),
        ];

        let default_material =
            Material::default_material(&gpu.device, &gpu.queue, mat_layout);

        // ── Build scene graph ───────────────────────────────────
        let mut scene = SceneGraph::new();

        let maze_config = MazeMeshConfig {
            wall_mesh: 0,
            floor_mesh: 1,
            wall_material: 0,
            floor_material: 1,
            ghost_house_material: 2,
            ghost_door_material: 3,
        };
        let maze_scene = maze_renderer::build_maze_scene(&mut scene, &maze, &maze_config);

        // Pac-Man entity node (sphere).
        let pacman_node = scene.add_child(scene.root(), Transform::IDENTITY);
        scene.node_mut(pacman_node).mesh = Some(2);
        scene.node_mut(pacman_node).material = Some(4);

        // Ghost entity nodes (spheres with ghost-colored materials).
        let ghost_mat_indices = [5usize, 6, 7, 8];
        let mut ghost_nodes = [pacman_node; 4];
        for (i, &mat_idx) in ghost_mat_indices.iter().enumerate() {
            let node = scene.add_child(scene.root(), Transform::IDENTITY);
            scene.node_mut(node).mesh = Some(2);
            scene.node_mut(node).material = Some(mat_idx);
            ghost_nodes[i] = node;
        }

        scene.update_world_matrices();

        // ── Camera ──────────────────────────────────────────────
        let mut camera = default_camera();
        camera.aspect = size.width as f32 / size.height.max(1) as f32;

        // ── UI overlay ──────────────────────────────────────────
        let ui = UiOverlay::new(
            &gpu.device,
            &gpu.queue,
            gpu.format(),
            size.width,
            size.height,
        );

        // ── Create game state ───────────────────────────────────
        let game = GameState::new(maze);

        self.render = Some(RenderState {
            gpu,
            depth,
            renderer,
            lights,
            scene,
            camera,
            materials,
            meshes,
            default_material,
            _maze_scene: maze_scene,
            ui,
            pacman_node,
            ghost_nodes,
        });

        self.game = Some(game);

        window.request_redraw();
        self.window = Some(window);
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
                        render.ui.resize(&render.gpu.queue, width, height);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.input.begin_frame();
                self.time.tick();

                if let Some(game) = &mut self.game {
                    // Process input.
                    process_input(&self.input, game);

                    // Handle restart on GAME OVER.
                    if game.phase == GamePhase::GameOver
                        && game.phase_timer >= GAME_OVER_DURATION
                        && self.input.key_pressed(KeyCode::Space)
                    {
                        game.reset_game();
                    }

                    // Fixed-timestep physics updates.
                    let dt = self.time.fixed_dt() as f32;
                    while self.time.should_do_fixed_update() {
                        fixed_update(game, dt);
                    }

                    // Sync scene graph and render.
                    let alpha = self.time.alpha();
                    if let Some(render) = &mut self.render {
                        sync_scene(render, game, alpha);

                        match render_frame(render, game) {
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
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

// ── Entry point ─────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    log::info!("Pac-Man 3D starting...");

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}
