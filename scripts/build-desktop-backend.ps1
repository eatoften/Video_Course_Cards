param(
    [string]$TargetTriple = "x86_64-pc-windows-msvc"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$BackendDir = Join-Path $RepoRoot "backend"
$BinariesDir = Join-Path $RepoRoot "frontend\src-tauri\binaries"
$BackendName = "video-course-cards-backend"
$SidecarName = "$BackendName-$TargetTriple.exe"

New-Item -ItemType Directory -Force -Path $BinariesDir | Out-Null

Push-Location $BackendDir
try {
    uv run pyinstaller `
        --noconfirm `
        --clean `
        --onefile `
        --console `
        --name $BackendName `
        --distpath "dist\desktop" `
        --workpath "build\pyinstaller" `
        --specpath "build\pyinstaller_specs" `
        --hidden-import "app.main" `
        --collect-submodules "app" `
        --collect-submodules "uvicorn" `
        --collect-submodules "pptx" `
        --collect-submodules "docx" `
        --collect-submodules "pypdf" `
        "app\desktop_server.py"
}
finally {
    Pop-Location
}

$BuiltExe = Join-Path $BackendDir "dist\desktop\$BackendName.exe"
$SidecarExe = Join-Path $BinariesDir $SidecarName

if (-not (Test-Path -LiteralPath $BuiltExe)) {
    throw "PyInstaller did not produce expected executable: $BuiltExe"
}

Copy-Item -LiteralPath $BuiltExe -Destination $SidecarExe -Force

Write-Host "Built backend executable:"
Write-Host "  $BuiltExe"
Write-Host "Copied Tauri sidecar candidate:"
Write-Host "  $SidecarExe"
