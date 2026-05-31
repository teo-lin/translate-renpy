# Translation Quality Benchmark Script (Thin Wrapper)
#
# Default - prompts for game, scores all model columns from an existing parsed YAML:
#   .\9-benchmark.ps1
#   .\9-benchmark.ps1 -Parsed "games\Example Uncensored\game\tl\romanian\Cell02_Wonderland.parsed.yaml"
#
# Re-translate mode - loads a model and re-translates the benchmark set (slow):
#   .\9-benchmark.ps1 -Retranslate [-ModelName aya23] [-Model N] [-Lang ro] [-Yes]

param(
    [string]$Parsed,                                  # Optional: skip game selector, use this file directly
    [string]$BenchmarkFile,                           # Reference YAML (auto-detected from language if omitted)
    [switch]$Retranslate,                             # Use old re-translate mode instead of score-parsed
    [int]$Model = 0,                                  # (Retranslate) model number, 0 = prompt
    [string]$ModelName,                               # (Retranslate) model key, e.g. "aya23"
    [string]$GlossaryFile,                            # (Retranslate) glossary override
    [string]$Lang = "ro",                             # (Retranslate) language code
    [switch]$Yes                                      # (Retranslate) skip confirmation
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$pythonExe = Join-Path $scriptDir "venv\Scripts\python.exe"
$benchmarkScript = Join-Path $scriptDir "scripts\benchmark.py"

$torchLibPath = Join-Path $scriptDir "venv\Lib\site-packages\torch\lib"
if (Test-Path $torchLibPath) { $env:PATH += ";$torchLibPath" }

$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path $pythonExe)) {
    Write-Host "ERROR: Python executable not found at $pythonExe" -ForegroundColor Red; exit 1
}
if (-not (Test-Path $benchmarkScript)) {
    Write-Host "ERROR: Benchmark script not found at $benchmarkScript" -ForegroundColor Red; exit 1
}

if ($Retranslate) {
    # -- Re-translate mode: load a model and score against benchmark data -----
    $scriptArgs = @("orchestrate", "--lang", $Lang)
    if ($BenchmarkFile) { $scriptArgs += @("--benchmark", $BenchmarkFile) }
    if ($ModelName)     { $scriptArgs += @("--model-key",    $ModelName)  }
    if ($Model -gt 0)   { $scriptArgs += @("--model-number", $Model)      }
    if ($GlossaryFile)  { $scriptArgs += @("--glossary",     $GlossaryFile) }
    if ($Yes)           { $scriptArgs += "-y"                              }
} else {
    # -- Default: score existing translations from a parsed YAML --------------
    $scriptArgs = @("score-parsed")
    if ($Parsed)        { $scriptArgs += @("--parsed",    $Parsed)        }
    if ($BenchmarkFile) { $scriptArgs += @("--benchmark", $BenchmarkFile) }
}

& $pythonExe $benchmarkScript $scriptArgs
exit $LASTEXITCODE
