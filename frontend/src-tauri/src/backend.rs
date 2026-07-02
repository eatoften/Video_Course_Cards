use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::Command,
    sync::Mutex,
    time::{Duration, Instant},
};

use serde::Serialize;
use tauri::{AppHandle, State};
use tauri_plugin_shell::{process::CommandChild, ShellExt};

const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: u16 = 8001;
const SIDECAR_NAME: &str = "video-course-cards-backend";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(1);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(45);
const POLL_INTERVAL: Duration = Duration::from_millis(300);
const APP_DATA_DIR_NAME: &str = "Video Course Cards";

pub struct BackendState {
    child: Mutex<Option<CommandChild>>,
    last_message: Mutex<String>,
}

impl BackendState {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            last_message: Mutex::new("Backend not started.".to_string()),
        }
    }

    fn set_message(&self, message: impl Into<String>) {
        if let Ok(mut last_message) = self.last_message.lock() {
            *last_message = message.into();
        }
    }

    fn message(&self) -> String {
        self.last_message
            .lock()
            .map(|message| message.clone())
            .unwrap_or_else(|_| "Backend state unavailable.".to_string())
    }
}

impl Drop for BackendState {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            if let Some(process) = child.take() {
                let _ = process.kill();
            }
        }
    }
}

#[derive(Clone, Serialize)]
pub struct BackendStatus {
    ready: bool,
    mode: String,
    message: String,
}

#[tauri::command]
pub fn get_backend_status(state: State<'_, BackendState>) -> BackendStatus {
    let ready = is_backend_ready();
    let mode = if state
        .child
        .lock()
        .map(|child| child.is_some())
        .unwrap_or(false)
    {
        "sidecar"
    } else if ready {
        "external"
    } else {
        "stopped"
    };

    BackendStatus {
        ready,
        mode: mode.to_string(),
        message: state.message(),
    }
}

#[tauri::command]
pub fn ensure_backend(
    app: AppHandle,
    state: State<'_, BackendState>,
) -> Result<BackendStatus, String> {
    ensure_backend_inner(&app, &state, false)
}

#[tauri::command]
pub fn restart_backend(
    app: AppHandle,
    state: State<'_, BackendState>,
) -> Result<BackendStatus, String> {
    stop_owned_backend(&state)?;
    kill_processes_on_backend_port()?;
    ensure_backend_inner(&app, &state, true)
}

#[tauri::command]
pub fn stop_backend(state: State<'_, BackendState>) -> Result<BackendStatus, String> {
    stop_owned_backend(&state)?;
    state.set_message("Stopped owned backend sidecar.");

    Ok(get_backend_status(state))
}

fn ensure_backend_inner(
    app: &AppHandle,
    state: &BackendState,
    force_spawn: bool,
) -> Result<BackendStatus, String> {
    if !force_spawn && is_backend_ready() {
        let mode = if has_owned_backend(state) {
            "sidecar"
        } else {
            "external"
        };
        let message = format!("Backend ready at {}.", backend_base_url());
        state.set_message(message.clone());
        return Ok(BackendStatus {
            ready: true,
            mode: mode.to_string(),
            message,
        });
    }

    if has_owned_backend(state) {
        state.set_message("Waiting for owned backend sidecar to become ready.");
        return wait_for_ready_status(state, "sidecar");
    }

    let occupied_pids = listening_pids_on_backend_port()?;
    if !occupied_pids.is_empty() {
        let message = format!(
            "Port {} is occupied by process(es): {}. Closing them before starting backend.",
            BACKEND_PORT,
            occupied_pids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        state.set_message(message);

        for pid in occupied_pids {
            kill_process(pid)?;
        }

        std::thread::sleep(Duration::from_secs(1));
    }

    start_sidecar(app, state)?;
    wait_for_ready_status(state, "sidecar")
}

fn start_sidecar(app: &AppHandle, state: &BackendState) -> Result<(), String> {
    state.set_message("Starting backend sidecar.");
    let app_data_dir = app_data_dir()?;
    let log_dir = app_data_dir.join("logs");
    std::fs::create_dir_all(&log_dir)
        .map_err(|error| format!("Failed to create backend log directory: {error}"))?;
    let log_file = log_dir.join("backend.log");
    let data_dir = app_data_dir.to_string_lossy().to_string();
    let log_file = log_file.to_string_lossy().to_string();

    let sidecar = app
        .shell()
        .sidecar(SIDECAR_NAME)
        .map_err(|error| format!("Failed to prepare backend sidecar: {error}"))?
        .env("VCC_DESKTOP", "1")
        .env("VCC_DATA_DIR", &data_dir)
        .env("VCC_BACKEND_LOG_FILE", &log_file)
        .args([
            "--host",
            BACKEND_HOST,
            "--port",
            &BACKEND_PORT.to_string(),
            "--no-reuse-existing",
            "--desktop",
            "--log-file",
            &log_file,
        ]);

    let (mut receiver, child) = sidecar
        .spawn()
        .map_err(|error| format!("Failed to spawn backend sidecar: {error}"))?;

    {
        let mut state_child = state
            .child
            .lock()
            .map_err(|_| "Backend process lock is poisoned.".to_string())?;
        *state_child = Some(child);
    }

    tauri::async_runtime::spawn(async move {
        while let Some(event) = receiver.recv().await {
            println!("backend sidecar event: {event:?}");
        }
    });

    Ok(())
}

fn app_data_dir() -> Result<PathBuf, String> {
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        return Ok(PathBuf::from(local_app_data).join(APP_DATA_DIR_NAME));
    }

    let current_dir = std::env::current_dir()
        .map_err(|error| format!("Failed to determine current directory: {error}"))?;

    Ok(current_dir.join(".video-course-cards"))
}

fn stop_owned_backend(state: &BackendState) -> Result<(), String> {
    let mut child = state
        .child
        .lock()
        .map_err(|_| "Backend process lock is poisoned.".to_string())?;

    if let Some(process) = child.take() {
        process
            .kill()
            .map_err(|error| format!("Failed to stop backend sidecar: {error}"))?;
    }

    Ok(())
}

fn wait_for_ready_status(state: &BackendState, mode: &str) -> Result<BackendStatus, String> {
    let started_at = Instant::now();

    while started_at.elapsed() < STARTUP_TIMEOUT {
        if is_backend_ready() {
            let message = format!("Backend ready at {}.", backend_base_url());
            state.set_message(message.clone());
            return Ok(BackendStatus {
                ready: true,
                mode: mode.to_string(),
                message,
            });
        }

        std::thread::sleep(POLL_INTERVAL);
    }

    let message = format!(
        "Backend did not become ready within {} seconds.",
        STARTUP_TIMEOUT.as_secs()
    );
    state.set_message(message.clone());

    Err(message)
}

fn has_owned_backend(state: &BackendState) -> bool {
    state
        .child
        .lock()
        .map(|child| child.is_some())
        .unwrap_or(false)
}

fn backend_base_url() -> String {
    format!("http://{}:{}", BACKEND_HOST, BACKEND_PORT)
}

fn is_backend_ready() -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], BACKEND_PORT));
    let mut stream = match TcpStream::connect_timeout(&address, HEALTH_TIMEOUT) {
        Ok(stream) => stream,
        Err(_) => return false,
    };

    let _ = stream.set_read_timeout(Some(HEALTH_TIMEOUT));
    let _ = stream.set_write_timeout(Some(HEALTH_TIMEOUT));

    let request = format!(
        "GET /health HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
        BACKEND_HOST, BACKEND_PORT
    );

    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }

    let mut response = String::new();
    if stream.read_to_string(&mut response).is_err() {
        return false;
    }

    response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200")
}

fn listening_pids_on_backend_port() -> Result<Vec<u32>, String> {
    let output = Command::new("netstat")
        .args(["-ano", "-p", "tcp"])
        .output()
        .map_err(|error| format!("Failed to inspect TCP ports: {error}"))?;

    if !output.status.success() {
        return Err("netstat failed while inspecting backend port.".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let port_suffix = format!(":{BACKEND_PORT}");
    let current_pid = std::process::id();
    let mut pids = Vec::new();

    for line in stdout.lines() {
        let columns = line.split_whitespace().collect::<Vec<_>>();

        if columns.len() < 5 {
            continue;
        }

        let local_address = columns[1];
        let state = columns[3];
        let pid = columns[4].parse::<u32>().ok();

        if state.eq_ignore_ascii_case("LISTENING")
            && local_address.ends_with(&port_suffix)
            && pid.is_some()
        {
            let pid = pid.unwrap();
            if pid != current_pid && !pids.contains(&pid) {
                pids.push(pid);
            }
        }
    }

    Ok(pids)
}

fn kill_processes_on_backend_port() -> Result<(), String> {
    for pid in listening_pids_on_backend_port()? {
        kill_process(pid)?;
    }

    std::thread::sleep(Duration::from_secs(1));
    Ok(())
}

fn kill_process(pid: u32) -> Result<(), String> {
    let output = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .output()
        .map_err(|error| format!("Failed to run taskkill for PID {pid}: {error}"))?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!(
        "Failed to close process {pid} on backend port: {}",
        stderr.trim()
    ))
}
