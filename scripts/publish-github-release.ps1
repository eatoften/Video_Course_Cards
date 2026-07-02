param(
    [Parameter(Mandatory = $true)]
    [string]$Tag,
    [string]$Title = "Video Course Cards $Tag",
    [string]$Notes = "Local-first desktop demo release.",
    [string]$ArtifactGlob = "frontend\src-tauri\target\release\bundle\**\*.exe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI `gh` is required to publish a release."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

Push-Location $RepoRoot
try {
    $artifacts = Get-ChildItem -Path $ArtifactGlob -File -ErrorAction SilentlyContinue

    if (-not $artifacts) {
        throw "No release artifacts matched: $ArtifactGlob"
    }

    $artifactPaths = @($artifacts | ForEach-Object { $_.FullName })

    gh release create $Tag `
        @artifactPaths `
        --title $Title `
        --notes $Notes `
        --draft
}
finally {
    Pop-Location
}
