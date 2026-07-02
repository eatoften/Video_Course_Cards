mod backend;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(backend::BackendState::new())
        .invoke_handler(tauri::generate_handler![
            backend::ensure_backend,
            backend::get_backend_status,
            backend::restart_backend,
            backend::stop_backend
        ])
        .run(tauri::generate_context!())
        .expect("error while running Video Course Cards");
}
