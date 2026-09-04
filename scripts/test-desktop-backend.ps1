param(
    [int]$Port = 0,
    [int]$TimeoutSeconds = 45
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$BackendExe = Join-Path $RepoRoot "frontend\src-tauri\binaries\citefold-backend-x86_64-pc-windows-msvc.exe"
$SmokeDir = Join-Path $RepoRoot ".desktop-smoke"
$LogFile = Join-Path $SmokeDir "backend-smoke.log"
$StdoutFile = Join-Path $SmokeDir "backend-smoke.stdout.log"
$StderrFile = Join-Path $SmokeDir "backend-smoke.stderr.log"
$DataDir = Join-Path $SmokeDir "data"
$InstanceToken = [Guid]::NewGuid().ToString("N")
$PreviousInstanceToken = [Environment]::GetEnvironmentVariable(
    "CITEFOLD_BACKEND_INSTANCE_TOKEN",
    "Process"
)
$PreviousDesktop = [Environment]::GetEnvironmentVariable(
    "CITEFOLD_DESKTOP",
    "Process"
)
$PreviousDataDir = [Environment]::GetEnvironmentVariable(
    "CITEFOLD_DATA_DIR",
    "Process"
)
$PreviousLogFile = [Environment]::GetEnvironmentVariable(
    "CITEFOLD_BACKEND_LOG_FILE",
    "Process"
)

if (-not (Test-Path -LiteralPath $BackendExe)) {
    throw "Backend sidecar executable not found. Run scripts\build-desktop-backend.ps1 first."
}

if ($Port -eq 0) {
    $portProbe = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        0
    )
    try {
        $portProbe.Start()
        $Port = ([System.Net.IPEndPoint]$portProbe.LocalEndpoint).Port
    }
    finally {
        $portProbe.Stop()
    }
}

New-Item -ItemType Directory -Force -Path $SmokeDir | Out-Null
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
Remove-Item -LiteralPath $LogFile -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $StdoutFile -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $StderrFile -Force -ErrorAction SilentlyContinue

$env:CITEFOLD_DESKTOP = "1"
$env:CITEFOLD_DATA_DIR = $DataDir
$env:CITEFOLD_BACKEND_LOG_FILE = $LogFile
$env:CITEFOLD_BACKEND_INSTANCE_TOKEN = $InstanceToken

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
    -RedirectStandardOutput $StdoutFile `
    -RedirectStandardError $StderrFile `
    -PassThru

try {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $healthUrl = "http://127.0.0.1:$Port/health"

    while ((Get-Date) -lt $deadline) {
        if ($process.HasExited) {
            $stderr = if (Test-Path -LiteralPath $StderrFile) {
                (Get-Content -LiteralPath $StderrFile -Raw).Trim()
            }
            else {
                ""
            }
            throw (
                "Backend sidecar exited early with code $($process.ExitCode). " +
                "Log: $LogFile" +
                $(if ($stderr) { "`nStderr: $stderr" } else { "" })
            )
        }

        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 2
            $health = $response.Content | ConvertFrom-Json
            if (
                $response.StatusCode -eq 200 -and
                $health.application_id -eq "citefold" -and
                $health.instance_token -eq $InstanceToken
            ) {
                Write-Host "Backend sidecar smoke test passed: $healthUrl"
                Write-Host "Log file: $LogFile"
                # Return from this script instead of terminating a parent
                # build-windows-installer.ps1 process invoked in the same host.
                return
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
    [Environment]::SetEnvironmentVariable(
        "CITEFOLD_BACKEND_INSTANCE_TOKEN",
        $PreviousInstanceToken,
        "Process"
    )
    [Environment]::SetEnvironmentVariable(
        "CITEFOLD_DESKTOP",
        $PreviousDesktop,
        "Process"
    )
    [Environment]::SetEnvironmentVariable(
        "CITEFOLD_DATA_DIR",
        $PreviousDataDir,
        "Process"
    )
    [Environment]::SetEnvironmentVariable(
        "CITEFOLD_BACKEND_LOG_FILE",
        $PreviousLogFile,
        "Process"
    )
}
