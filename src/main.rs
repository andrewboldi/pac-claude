fn main() {
    env_logger::init();
    log::info!("Pac-Man 3D starting...");
    pac_window::run(pac_window::WindowConfig::default());
}
