# Translation Model Comparison Script (Thin Wrapper)
# Compares installed models by translating the same content
#
# Run all installed models (default):
#   .\8-compare.ps1
#
# Run only specific models (adds missing keys, skips existing):
#   .\8-compare.ps1 -Models "euroLLM22b,opusTCBig,nllb1300"

param(
    [string]$GameName,
    [string]$Language = "ro",
    [string]$Models      # Comma-separated model keys; runs all installed if omitted
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$pythonCmd = Join-Path $scriptDir "venv\Scripts\python.exe"
$compareScript = Join-Path $scriptDir "scripts\compare.py"

if (-not (Test-Path $pythonCmd)) {
    Write-Host "ERROR: Python executable not found at $pythonCmd" -ForegroundColor Red
    exit 1
}

# Build orchestrator args; omit --game to fall through to the interactive picker
$scriptArgs = @("orchestrate", "--language", $Language)
if ($GameName) { $scriptArgs += @("--game",   $GameName) }
if ($Models)   { $scriptArgs += @("--models", $Models)   }

& $pythonCmd $compareScript $scriptArgs
exit $LASTEXITCODE
