param(
    [ValidateSet("install", "run")]
    [string]$Action = "run",
    [int]$Port = 5000
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path (Split-Path $projectRoot -Parent) -Parent
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found: $venvPython"
}

Push-Location $projectRoot
try {
    switch ($Action) {
        "install" {
            & $venvPython -m pip install -r requirements.txt
            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
        }
        "run" {
            $env:HOST = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
            $env:PORT = "$Port"
            & $venvPython app.py
            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
        }
    }
}
finally {
    Pop-Location
}
