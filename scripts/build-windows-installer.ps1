param(
    [ValidateSet("nsis", "msi")]
    [string]$Bundle = "nsis",
    [switch]$SkipBackendBuild,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$FrontendDir = Join-Path $RepoRoot "frontend"
$SidecarExe = Join-Path $FrontendDir "src-tauri\binaries\video-course-cards-backend-x86_64-pc-windows-msvc.exe"
$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"

if ((Test-Path -LiteralPath $CargoBin) -and $env:Path -notlike "*$CargoBin*") {
    $env:Path = "$env:Path;$CargoBin"
}

function Require-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found on PATH: $Name"
    }
}

Require-Command "uv"
Require-Command "npm.cmd"
Require-Command "cargo"
Require-Command "rustc"

if (-not $SkipBackendBuild) {
    Write-Host "Building backend sidecar..."
    & (Join-Path $ScriptDir "build-desktop-backend.ps1")
}

if (-not (Test-Path -LiteralPath $SidecarExe)) {
    throw "Backend sidecar missing: $SidecarExe"
}

if (-not $SkipSmokeTest) {
    Write-Host "Running backend sidecar smoke test..."
    & (Join-Path $ScriptDir "test-desktop-backend.ps1")
}

Push-Location $FrontendDir
try {
    Write-Host "Building Tauri $Bundle installer..."
    npm.cmd exec -- tauri build --bundles $Bundle --ci
}
finally {
    Pop-Location
}

$BundleDir = Join-Path $FrontendDir "src-tauri\target\release\bundle"

Write-Host ""
Write-Host "Installer build complete. Candidate artifacts:"
Get-ChildItem -Path $BundleDir -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in ".exe", ".msi" } |
    Select-Object FullName, Length |
    Format-Table -AutoSize
