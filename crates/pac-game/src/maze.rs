use serde::{Deserialize, Serialize};

/// Width of the classic Pac-Man maze in tiles.
pub const MAZE_WIDTH: usize = 28;
/// Height of the classic Pac-Man maze in tiles.
pub const MAZE_HEIGHT: usize = 31;

/// The type of each tile in the maze grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TileType {
    Empty,
    Wall,
    Pellet,
    PowerPellet,
    GhostHouse,
    GhostDoor,
    PlayerSpawn,
}

/// Raw JSON representation of the maze file.
#[derive(Debug, Deserialize)]
struct MazeJson {
    width: usize,
    height: usize,
    tiles: Vec<Vec<u8>>,
}

/// The maze grid data: a 28×31 array of tile types.
#[derive(Debug, Clone)]
pub struct MazeData {
    pub tiles: [[TileType; MAZE_WIDTH]; MAZE_HEIGHT],
}

impl MazeData {
    /// Load maze data from a JSON byte slice.
    ///
    /// The JSON format uses a `tiles` array of rows, where each row is an array
    /// of integer tile codes:
    ///   0 = Empty, 1 = Wall, 2 = Pellet, 3 = PowerPellet,
    ///   4 = GhostHouse, 5 = GhostDoor, 6 = PlayerSpawn
    pub fn from_json(data: &[u8]) -> Result<Self, MazeError> {
        let raw: MazeJson =
            serde_json::from_slice(data).map_err(|e| MazeError::Parse(e.to_string()))?;

        if raw.width != MAZE_WIDTH || raw.height != MAZE_HEIGHT {
            return Err(MazeError::Dimensions {
                width: raw.width,
                height: raw.height,
            });
        }

        if raw.tiles.len() != MAZE_HEIGHT {
            return Err(MazeError::RowCount(raw.tiles.len()));
        }

        let mut tiles = [[TileType::Empty; MAZE_WIDTH]; MAZE_HEIGHT];

        for (y, row) in raw.tiles.iter().enumerate() {
            if row.len() != MAZE_WIDTH {
                return Err(MazeError::ColumnCount { row: y, count: row.len() });
            }
            for (x, &code) in row.iter().enumerate() {
                tiles[y][x] = tile_from_code(code).ok_or(MazeError::InvalidTile {
                    row: y,
                    col: x,
                    code,
                })?;
            }
        }

        Ok(MazeData { tiles })
    }

    /// Get the tile type at the given grid position.
    /// Returns `None` if the coordinates are out of bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Option<TileType> {
        if x < MAZE_WIDTH && y < MAZE_HEIGHT {
            Some(self.tiles[y][x])
        } else {
            None
        }
    }

    /// Returns `true` if the tile at `(x, y)` is a wall.
    #[inline]
    pub fn is_wall(&self, x: usize, y: usize) -> bool {
        self.get(x, y) == Some(TileType::Wall)
    }
}

fn tile_from_code(code: u8) -> Option<TileType> {
    match code {
        0 => Some(TileType::Empty),
        1 => Some(TileType::Wall),
        2 => Some(TileType::Pellet),
        3 => Some(TileType::PowerPellet),
        4 => Some(TileType::GhostHouse),
        5 => Some(TileType::GhostDoor),
        6 => Some(TileType::PlayerSpawn),
        _ => None,
    }
}

/// Errors that can occur when loading maze data.
#[derive(Debug)]
pub enum MazeError {
    Parse(String),
    Dimensions { width: usize, height: usize },
    RowCount(usize),
    ColumnCount { row: usize, count: usize },
    InvalidTile { row: usize, col: usize, code: u8 },
}

impl std::fmt::Display for MazeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MazeError::Parse(e) => write!(f, "JSON parse error: {e}"),
            MazeError::Dimensions { width, height } => {
                write!(f, "expected {MAZE_WIDTH}x{MAZE_HEIGHT} maze, got {width}x{height}")
            }
            MazeError::RowCount(n) => {
                write!(f, "expected {MAZE_HEIGHT} rows, got {n}")
            }
            MazeError::ColumnCount { row, count } => {
                write!(f, "row {row}: expected {MAZE_WIDTH} columns, got {count}")
            }
            MazeError::InvalidTile { row, col, code } => {
                write!(f, "invalid tile code {code} at ({col}, {row})")
            }
        }
    }
}

impl std::error::Error for MazeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn classic_json() -> Vec<u8> {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/maze/classic.json"),
        )
        .expect("classic.json should exist")
    }

    #[test]
    fn load_classic_maze() {
        let data = classic_json();
        let maze = MazeData::from_json(&data).expect("should parse classic maze");

        // Top-left corner is a wall
        assert_eq!(maze.tiles[0][0], TileType::Wall);
        // Dimensions are correct
        assert_eq!(maze.tiles.len(), MAZE_HEIGHT);
        assert_eq!(maze.tiles[0].len(), MAZE_WIDTH);
    }

    #[test]
    fn get_out_of_bounds() {
        let data = classic_json();
        let maze = MazeData::from_json(&data).unwrap();
        assert_eq!(maze.get(MAZE_WIDTH, 0), None);
        assert_eq!(maze.get(0, MAZE_HEIGHT), None);
    }

    #[test]
    fn is_wall_checks() {
        let data = classic_json();
        let maze = MazeData::from_json(&data).unwrap();
        // The border of the classic maze is walls
        assert!(maze.is_wall(0, 0));
        assert!(maze.is_wall(MAZE_WIDTH - 1, 0));
    }

    #[test]
    fn wrong_dimensions_rejected() {
        let json = r#"{"width":10,"height":10,"tiles":[]}"#;
        let err = MazeData::from_json(json.as_bytes()).unwrap_err();
        assert!(matches!(err, MazeError::Dimensions { .. }));
    }

    #[test]
    fn wrong_row_count_rejected() {
        let json = r#"{"width":28,"height":31,"tiles":[]}"#;
        let err = MazeData::from_json(json.as_bytes()).unwrap_err();
        assert!(matches!(err, MazeError::RowCount(_)));
    }

    #[test]
    fn invalid_tile_code_rejected() {
        // Build a valid-shaped maze but with an invalid tile code (99)
        let mut rows = Vec::new();
        for _ in 0..MAZE_HEIGHT {
            rows.push(vec![0u8; MAZE_WIDTH]);
        }
        rows[0][0] = 99;
        let raw = serde_json::json!({
            "width": MAZE_WIDTH,
            "height": MAZE_HEIGHT,
            "tiles": rows,
        });
        let err = MazeData::from_json(raw.to_string().as_bytes()).unwrap_err();
        assert!(matches!(err, MazeError::InvalidTile { row: 0, col: 0, code: 99 }));
    }

    #[test]
    fn player_spawn_exists() {
        let data = classic_json();
        let maze = MazeData::from_json(&data).unwrap();
        let has_spawn = maze
            .tiles
            .iter()
            .any(|row| row.iter().any(|t| *t == TileType::PlayerSpawn));
        assert!(has_spawn, "classic maze should have a player spawn");
    }

    #[test]
    fn ghost_house_exists() {
        let data = classic_json();
        let maze = MazeData::from_json(&data).unwrap();
        let has_ghost_house = maze
            .tiles
            .iter()
            .any(|row| row.iter().any(|t| *t == TileType::GhostHouse));
        assert!(has_ghost_house, "classic maze should have a ghost house");
    }
}
