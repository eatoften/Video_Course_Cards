$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$frontendUrl = "http://127.0.0.1:5174"

try {
    $response = Invoke-WebRequest `
        -UseBasicParsing `
        -Uri $frontendUrl `
        -TimeoutSec 2

    if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
        Write-Host "Reusing existing Vite dev server at $frontendUrl"
        exit 0
    }
}
catch {
    Write-Host "No Vite dev server detected at $frontendUrl. Starting one."
}

npm.cmd run dev
