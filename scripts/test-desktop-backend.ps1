param(
    [int]$Port = 8765,
    [int]$TimeoutSeconds = 45
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$BackendExe = Join-Path $RepoRoot "frontend\src-tauri\binaries\video-course-cards-backend-x86_64-pc-windows-msvc.exe"
$SmokeDir = Join-Path $RepoRoot ".desktop-smoke"
$LogFile = Join-Path $SmokeDir "backend-smoke.log"
$DataDir = Join-Path $SmokeDir "data"

if (-not (Test-Path -LiteralPath $BackendExe)) {
    throw "Backend sidecar executable not found. Run scripts\build-desktop-backend.ps1 first."
}

New-Item -ItemType Directory -Force -Path $SmokeDir | Out-Null
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
Remove-Item -LiteralPath $LogFile -Force -ErrorAction SilentlyContinue

$env:VCC_DESKTOP = "1"
$env:VCC_DATA_DIR = $DataDir
$env:VCC_BACKEND_LOG_FILE = $LogFile

$process = Start-Process `
    -FilePath $BackendExe `
    -ArgumentList @(
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--desktop",
        "--no-reuse-existing",
        "--log-file", $LogFile
    ) `
    -WindowStyle Hidden `
    -PassThru

try {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $healthUrl = "http://127.0.0.1:$Port/health"

    while ((Get-Date) -lt $deadline) {
        if ($process.HasExited) {
            throw "Backend sidecar exited early with code $($process.ExitCode). Log: $LogFile"
        }

        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                Write-Host "Backend sidecar smoke test passed: $healthUrl"
                Write-Host "Log file: $LogFile"
                exit 0
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    throw "Backend sidecar did not become healthy within $TimeoutSeconds seconds. Log: $LogFile"
}
finally {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
}
