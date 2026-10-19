<#
.SYNOPSIS
    One-shot environment bootstrap for Windows PowerShell.

.DESCRIPTION
    Creates a Python 3.10 venv at .\.venv and installs the project (editable).
    Uses `uv` if available (10-100x faster); falls back to standard pip otherwise.

    IMPORTANT: TensorFlow 2.11 extracts deeply nested headers that exceed
    Windows' default 260-character path limit. If the install fails with
    "No such file or directory", you have two options:
      1. Enable long paths system-wide (needs admin):
         Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
             LongPathsEnabled 1 -Type DWord
      2. Create the venv outside the repo with -VenvDir:
         .\scripts\setup.ps1 -VenvDir "C:\Users\$env:USERNAME\tfg310"

.PARAMETER PythonVersion
    Python version to install (default: 3.10).

.PARAMETER VenvDir
    Where to create the venv (default: .\.venv next to the repo).

.PARAMETER Extras
    Optional extras to install (dev).

.EXAMPLE
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -Extras dev
    .\scripts\setup.ps1 -VenvDir "C:\Users\$env:USERNAME\tfg310"
#>

[CmdletBinding()]
param(
    [string]$PythonVersion = "3.10",
    [string]$VenvDir = ".venv",
    [string[]]$Extras = @()
)

$ErrorActionPreference = "Stop"

# --- locate repo root so the script works from any cwd ---------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

# --- ensure uv is available ------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "==> 'uv' not found. Installing locally (no admin needed)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "==> Ensuring Python $PythonVersion is available via uv..."
uv python install $PythonVersion

# --- create venv -----------------------------------------------------------
if (Test-Path $VenvDir) {
    Write-Host "==> Reusing existing venv at $VenvDir"
} else {
    Write-Host "==> Creating venv at $VenvDir (Python $PythonVersion)"
    uv venv --python $PythonVersion $VenvDir
}

# --- assemble install spec -------------------------------------------------
$spec = "."
if ($Extras.Count -gt 0) {
    $joined = ($Extras -join ",")
    $spec = ".[$joined]"
}

Write-Host "==> Installing $spec (editable)"
$venvPython = Join-Path $VenvDir "Scripts\python.exe"
uv pip install --python $venvPython -e $spec

Write-Host ""
Write-Host "==> Done."
Write-Host ""
Write-Host "Activate the venv with:"
Write-Host "    $VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then verify imports:"
Write-Host '    python -c "from src.activity_data import get_time_setup; print(get_time_setup(''WISDM''))"'
Write-Host ""
Write-Host "Dataset path defaults to .\data\WISDM\ (override with `$env:TFG_DATA_DIR)."
