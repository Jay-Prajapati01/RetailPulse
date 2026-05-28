param(
    [switch]$Build
)

Set-Location (Split-Path -Parent $PSScriptRoot)

if ($Build) {
    docker compose -f docker/docker-compose.yml up --build -d
} else {
    docker compose -f docker/docker-compose.yml up -d
}

Write-Host "RetailPulse stack started."
Write-Host "Streamlit: http://localhost:8501"
Write-Host "MLflow:    http://localhost:5000"
Write-Host "Airflow:   http://localhost:8080"
