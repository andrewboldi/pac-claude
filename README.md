# 3D Pac-Man Game Implementation Plan

## Overview
Build a 3D rendering engine from scratch in Rust (using wgpu/winit/glam), then implement a 3D Pac-Man game on top of it. Work delegated to polecats in the `pac` rig.

## Project Structure

```
pac/
├── Cargo.toml                    # Workspace manifest
├── assets/
│   ├── shaders/                  # WGSL shaders
│   │   ├── basic.wgsl           # Simple unlit
│   │   ├── phong.wgsl           # Phong lighting
│   │   └── ui.wgsl              # 2D UI overlay
│   └── maze/classic.json        # Classic maze layout
├── crates/
│   ├── pac-math/                # Math utilities (glam + extras)
│   │   └── src/ (transform.rs, bounds.rs, interpolation.rs)
│   ├── pac-render/              # 3D rendering engine
│   │   └── src/ (context.rs, pipeline.rs, buffer.rs, texture.rs,
│   │            mesh.rs, camera.rs, material.rs, light.rs,
│   │            scene.rs, renderer.rs)
│   ├── pac-window/              # Window/input layer
│   │   └── src/ (window.rs, input.rs, time.rs)
│   └── pac-game/                # Game logic
│       └── src/ (maze.rs, pacman.rs, ghost.rs, ai/*.rs,
│                pellet.rs, collision.rs, state.rs, ui.rs)
└── src/main.rs                  # Entry point
```

## Key Dependencies
- `wgpu = "23"` - GPU abstraction
- `winit = "0.30"` - Window/input
- `glam = "0.29"` - Math
- `bytemuck = "1.14"` - GPU data casting
- `image = "0.25"` - Texture loading
- `serde/serde_json` - Maze data

---

## Phase 1: Foundation (6 tasks)

| Task | Description | Deps | Parallel |
|------|-------------|------|----------|
| **1.1** | Workspace setup - Cargo.toml, crate stubs | - | Yes |
| **1.2** | Window creation - winit EventLoop, resize | 1.1 | - |
| **1.3** | Input system - keyboard/mouse state tracking | 1.2 | Yes w/1.4 |
| **1.4** | Time management - delta time, fixed timestep | 1.1 | Yes w/1.3 |
| **1.5** | GPU context - wgpu Instance/Surface/Device/Queue | 1.2 | - |
| **1.6** | Basic triangle render - validate GPU pipeline | 1.5 | - |

---

## Phase 2: 3D Primitives (6 tasks)

| Task | Description | Deps | Parallel |
|------|-------------|------|----------|
| **2.1** | Vertex/Buffer abstractions - VertexBuffer, UniformBuffer | 1.6 | Yes w/2.2 |
| **2.2** | Transform component - position/rotation/scale, matrices | 1.1 | Yes w/2.1 |
| **2.3** | Mesh primitives - cube, sphere, plane, cylinder | 2.1 | Yes w/2.4 |
| **2.4** | Camera system - perspective, view/proj matrices, controls | 2.2 | Yes w/2.3 |
| **2.5** | Pipeline abstraction - shader loading, bind groups | 2.1,2.3 | - |
| **2.6** | Depth buffer - depth testing, 3D occlusion | 2.5 | - |

---

## Phase 3: Lighting & Materials (5 tasks)

| Task | Description | Deps | Parallel |
|------|-------------|------|----------|
| **3.1** | Phong shader - ambient/diffuse/specular in WGSL | 2.6 | Yes w/3.2 |
| **3.2** | Texture loading - PNG via image crate, samplers | 2.5 | Yes w/3.1 |
| **3.3** | Material system - diffuse color/texture, specular | 3.1,3.2 | - |
| **3.4** | Light system - PointLight, DirectionalLight uniforms | 3.1 | Yes w/3.5 |
| **3.5** | Textured mesh rendering - UV sampling in shader | 3.2,3.3 | Yes w/3.4 |

---

## Phase 4: Scene Management (4 tasks)

| Task | Description | Deps | Parallel |
|------|-------------|------|----------|
| **4.1** | Scene graph - SceneNode hierarchy, world matrices | 3.3 | Yes w/4.2 |
| **4.2** | Instanced rendering - for 240+ pellets | 2.3 | Yes w/4.1 |
| **4.3** | Render orchestrator - batch by material, draw submission | 4.1,4.2 | - |
| **4.4** | Asset manager - mesh/material/texture handles | 4.3 | - |

---

## Phase 5: Game Core (6 tasks)

| Task | Description | Deps | Parallel |
|------|-------------|------|----------|
| **5.1** | Maze data structure - 28x31 grid, tile types, JSON load | 1.1 | Yes w/5.2 |
| **5.2** | AABB collision - intersection, containment tests | 1.1 | Yes w/5.1 |
| **5.3** | Maze 3D generation - walls, floor, ghost house geometry | 5.1,4.3 | - |
| **5.4** | Pac-Man entity - grid movement, direction queue, collision | 5.2,5.3 | Yes w/5.5 |
| **5.5** | Pellet system - spawn from maze, collection, instanced draw | 4.2,5.1 | Yes w/5.4 |
| **5.6** | Game collision - Pac-Man vs walls/pellets/ghosts | 5.4,5.5 | - |

---

## Phase 6: Ghost AI (5 tasks)

| Task | Description | Deps | Parallel |
|------|-------------|------|----------|
| **6.1** | Ghost base entity - state machine (Chase/Scatter/Frightened/Eaten) | 5.4 | - |
| **6.2** | Ghost mode controller - scatter/chase timer, frightened mode | 6.1 | Yes w/6.3-6.6 |
| **6.3** | Blinky AI - direct chase, Cruise Elroy | 6.1 | Yes |
| **6.4** | Pinky AI - 4 tiles ahead ambush | 6.1 | Yes |
| **6.5** | Inky AI - fickle, uses Blinky's position | 6.3 | After 6.3 |
| **6.6** | Clyde AI - shy/random, retreat when close | 6.1 | Yes |

---

## Phase 7: Polish (5 tasks)

| Task | Description | Deps | Parallel |
|------|-------------|------|----------|
| **7.1** | Game state machine - menu, playing, paused, game over | 6.2 | Yes w/7.2,7.3 |
| **7.2** | UI overlay - score, lives, READY!, GAME OVER | 3.2 | Yes w/7.1,7.3 |
| **7.3** | Animations - mouth cycle, ghost float, death | 5.4,6.1 | Yes w/7.1,7.2 |
| **7.4** | Sound hooks - event system for audio integration | 7.1 | Yes |
| **7.5** | Final integration - main loop, fixed timestep, shutdown | ALL | - |

---

## Parallelization Strategy

**Sprint 1**: Tasks 1.1-1.6 (sequential foundation)
**Sprint 2**: 2.1+2.2 (parallel), then 2.3+2.4 (parallel), then 2.5, 2.6
**Sprint 3**: 3.1+3.2 (parallel), then 3.3, then 3.4+3.5 (parallel)
**Sprint 4**: 4.1+4.2 (parallel) + 5.1+5.2 (parallel), then 4.3, 4.4, 5.3
**Sprint 5**: 5.4+5.5 (parallel), then 5.6
**Sprint 6**: 6.1, then 6.2+6.3+6.4+6.6 (parallel), then 6.5
**Sprint 7**: 7.1+7.2+7.3+7.4 (parallel), then 7.5

---

## Critical Files

- `crates/pac-render/src/context.rs` - GPU init, everything depends on this
- `crates/pac-render/src/mesh.rs` - All 3D primitives
- `assets/shaders/phong.wgsl` - Main lighting shader
- `crates/pac-game/src/maze.rs` - Level layout + collision
- `crates/pac-game/src/ghost.rs` - Ghost base + state machine

---

## Verification

1. **Phase 1-2**: Run binary, see spinning textured cube with depth
2. **Phase 3**: Cube with Phong shading responding to light
3. **Phase 4**: Multiple cubes in scene graph hierarchy
4. **Phase 5**: 3D maze rendered, Pac-Man moves through corridors
5. **Phase 6**: All 4 ghosts chase with correct AI behaviors
6. **Phase 7**: Complete game loop - start, play, die, game over, restart

Run: `cargo run` from pac rig root

---

## Delegation Plan

37 total tasks. Create beads for each, then sling to polecats:
- Phase 1-2: Sequential foundation (1 polecat initially)
- Phase 3+: Parallel polecats (2-4) for independent tasks
- Each bead includes: task description, files to create/modify, acceptance criteria
