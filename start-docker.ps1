param(
    [ValidateSet("build", "up", "down", "ps", "logs", "smoke", "shell")]
    [string]$Action = "up",
    [switch]$Gpu
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dockerConfig = Join-Path $scriptRoot ".docker-config"
if (-not (Test-Path $dockerConfig)) {
    New-Item -ItemType Directory -Path $dockerConfig | Out-Null
}
$env:DOCKER_CONFIG = $dockerConfig

$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($docker) {
    $dockerExe = $docker.Source
} else {
    $dockerExe = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
}

if (-not (Test-Path $dockerExe)) {
    throw "Docker executable not found. Install Docker Desktop first."
}

$composeArgs = @()
if ($Gpu) {
    $composeArgs += @("-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml")
}

switch ($Action) {
    "build" {
        & $dockerExe compose @composeArgs build
    }
    "up" {
        & $dockerExe compose @composeArgs up -d --build
    }
    "down" {
        & $dockerExe compose @composeArgs down
    }
    "ps" {
        & $dockerExe compose @composeArgs ps
    }
    "logs" {
        & $dockerExe compose @composeArgs logs -f --tail 200
    }
    "smoke" {
        & $dockerExe compose @composeArgs run --rm video-platform ai-smoke
    }
    "shell" {
        & $dockerExe compose @composeArgs run --rm video-platform shell
    }
}
