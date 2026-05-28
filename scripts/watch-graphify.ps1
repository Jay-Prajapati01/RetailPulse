param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

Set-Location $RepoRoot

if (-not (Get-Command graphify -ErrorAction SilentlyContinue)) {
    throw 'graphify is not available on PATH. Install it with: uv tool install graphifyy'
}

graphify watch .
