[CmdletBinding()]
param(
    [switch]$NoOpenPowerBI
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop was not found. Install it and run this script again."
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine is not running. Start Docker Desktop and wait until it is ready."
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from the local template."
}

Write-Host "Starting PostgreSQL..."
docker compose up -d --wait postgres
if ($LASTEXITCODE -ne 0) { throw "PostgreSQL failed to start." }

Write-Host "Building the analytics container..."
docker compose --profile tools build pipeline
if ($LASTEXITCODE -ne 0) { throw "The analytics image build failed." }

Write-Host "Downloading data and running the analytics pipeline..."
docker compose --profile tools run --rm pipeline python scripts/run_pipeline.py
if ($LASTEXITCODE -ne 0) { throw "The analytics pipeline failed." }

Write-Host "Verifying the deployment..."
docker compose --profile tools run --rm pipeline python scripts/verify_deployment.py
if ($LASTEXITCODE -ne 0) { throw "Deployment verification failed." }

$Pbip = Join-Path $ProjectRoot "powerbi\dashboard\RetailExperimentation.pbip"
Write-Host "Done. Dashboard: $Pbip"
Write-Host "PostgreSQL: localhost:5432; database retail_experiments; user retail."

if (-not $NoOpenPowerBI) {
    $Candidates = @(
        "$env:ProgramFiles\Microsoft Power BI Desktop\bin\PBIDesktop.exe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\PBIDesktop.exe"
    )
    $PowerBI = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($PowerBI) {
        Start-Process -FilePath $PowerBI -ArgumentList ('"' + $Pbip + '"')
    }
    else {
        Write-Warning "Power BI Desktop was not found. Install it and open the PBIP file manually."
    }
}
