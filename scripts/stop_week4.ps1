Set-Location (Split-Path -Parent $PSScriptRoot)

docker compose -f docker/docker-compose.yml down

Write-Host "RetailPulse stack stopped."
