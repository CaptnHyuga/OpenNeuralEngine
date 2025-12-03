<#
.SYNOPSIS
    One-Click SNN Launcher - Starts Aim tracking + Inference UI
.DESCRIPTION
    Professional launcher that handles all setup automatically:
    - Checks prerequisites (Python 3.11+, Docker)
    - Creates virtual environment
    - Installs dependencies
    - Starts Aim server via Docker
    - Launches inference UI
    - Opens browser automatically
.NOTES
    Version: 1.0
    Like Spotify or Unity Hub - just double-click and it works!
#>

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for pretty output
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "[*] $msg" -ForegroundColor Cyan }
function Write-Warning { param($msg) Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "[X] $msg" -ForegroundColor Red }

# Banner
Clear-Host
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   OpenNeuralEngine (ONE) - One-Click Launcher" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
# Get script directory
$ROOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT_DIR

Write-Info "Starting SNN application..."
Write-Host ""

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================
Write-Info "Step 1/6: Checking prerequisites..."

# Check Python
try {
    $pythonVersion = & python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -ge 3 -and $minor -ge 11) {
            Write-Success "Python $major.$minor found"
        } else {
            Write-Error "Python 3.11+ required. Found: $pythonVersion"
            Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
} catch {
    Write-Error "Python not found. Please install Python 3.11+"
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Docker (optional but recommended)
$dockerAvailable = $false
try {
    $null = & docker --version 2>&1
    $dockerAvailable = $true
    Write-Success "Docker found (Aim tracking will be enabled)"
} catch {
    Write-Warning "Docker not found - Aim tracking will be disabled"
    Write-Info "  Install Docker Desktop for full features: https://www.docker.com/products/docker-desktop"
}

Write-Host ""

# ============================================================================
# STEP 2: Setup Virtual Environment
# ============================================================================
Write-Info "Step 2/6: Setting up Python environment..."

$venvPath = Join-Path $ROOT_DIR ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Info "Creating virtual environment..."
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment"
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Success "Virtual environment created"
} else {
    Write-Success "Virtual environment exists"
}

# Activate virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
. $activateScript
Write-Success "Virtual environment activated"

Write-Host ""

# ============================================================================
# STEP 3: Install Dependencies
# ============================================================================
Write-Info "Step 3/6: Installing dependencies..."

# Check if already installed
$pipList = & pip list 2>&1
if ($pipList -notmatch "OpenNeuralEngine") {
    Write-Info "Installing SNN package (this may take a few minutes)..."
    Write-Warning "Note: Aim tracking requires Docker on Windows (Python package has compatibility issues)"
    
    # Install without output spam
    & pip install -e . --quiet 2>&1 | Out-Null
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install dependencies"
        Write-Info "Trying without aim package..."
        # Try installing just the core dependencies
        & pip install torch safetensors fastapi uvicorn pydantic tqdm evaluate transformers --quiet 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Installation failed. Check your internet connection."
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
    Write-Success "Dependencies installed"
} else {
    Write-Success "Dependencies already installed"
}

Write-Host ""

# ============================================================================
# STEP 4: Start Aim Server (if Docker available)
# ============================================================================
$aimEnabled = $false
if ($dockerAvailable) {
    Write-Info "Step 4/6: Starting Aim tracking server..."
    
    $aimProjectDir = Join-Path $ROOT_DIR ".aim_project"
    if (Test-Path $aimProjectDir) {
        Set-Location $aimProjectDir
        
        # Check if already running (accept legacy or current container names)
        $aimRunning = & docker ps --filter "name=snn-aim" --format "{{.Names}}" 2>&1
        if ($aimRunning -match "^snn-aim") {
            Write-Success "Aim server already running"
            $aimEnabled = $true
        } else {
            Write-Info "Starting Aim server container..."
            # Use Start-Process to avoid NativeCommandError on Compose progress output
            $prevErrPref = $ErrorActionPreference
            $ErrorActionPreference = "SilentlyContinue"
            try {
                $proc = Start-Process -FilePath "docker" -ArgumentList @("compose","up","-d") -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$env:TEMP\snn_aim_up.out" -RedirectStandardError "$env:TEMP\snn_aim_up.err"
                $composeExit = if ($proc) { $proc.ExitCode } else { $LASTEXITCODE }
            } finally {
                $ErrorActionPreference = $prevErrPref
            }
            if ($composeExit -eq 0) {
                # Wait for health check
                Write-Info "Waiting for Aim server to be ready..."
                $maxRetries = 30
                $retries = 0
                $aimReady = $false
                
                while ($retries -lt $maxRetries -and -not $aimReady) {
                    Start-Sleep -Seconds 1
                    try {
                        $response = Invoke-WebRequest -Uri "http://localhost:53800" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
                        if ($response.StatusCode -eq 200) {
                            $aimReady = $true
                        }
                    } catch {
                        $retries++
                    }
                }
                
                if ($aimReady) {
                    Write-Success "Aim server ready at http://localhost:53800"
                    $aimEnabled = $true
                } else {
                    Write-Warning "Aim server started but not responding yet"
                    Write-Info "  You can check http://localhost:53800 manually"
                }
            } else {
                Write-Warning "Failed to start Aim server - continuing without tracking"
                Write-Info "  See $env:TEMP\snn_aim_up.err for Docker compose logs"
            }
        }
        
        Set-Location $ROOT_DIR
    } else {
        Write-Warning ".aim_project folder not found - tracking disabled"
    }
} else {
    Write-Info "Step 4/6: Skipping Aim server (Docker not available)"
}

Write-Host ""

# ============================================================================
# STEP 5: Start Inference Server
# ============================================================================
Write-Info "Step 5/6: Starting inference server..."

# Set environment variables
if ($aimEnabled) {
    $env:AIM_TRACKING_URI = "aim://localhost:53800"
    $env:SNN_TRACKING_MODE = "enabled"
} else {
    $env:SNN_TRACKING_MODE = "disabled"
}

# Start inference server in background
Write-Info "Launching inference API (with multimodal support)..."
$inferenceJob = Start-Job -ScriptBlock {
    param($rootDir)
    Set-Location $rootDir
    . .venv\Scripts\Activate.ps1
    & python scripts\aim_native_inference.py --port 53801
} -ArgumentList $ROOT_DIR

# Wait for server to be ready
Write-Info "Waiting for inference server..."
$maxRetries = 60
$retries = 0
$serverReady = $false

while ($retries -lt $maxRetries -and -not $serverReady) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:53801/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $serverReady = $true
        }
    } catch {
        $retries++
    }
}

if ($serverReady) {
    Write-Success "Inference server ready at http://localhost:53801"
} else {
    Write-Error "Inference server failed to start in 60 seconds"
    Write-Info "Checking server output..."
    Receive-Job $inferenceJob
    Stop-Job $inferenceJob
    Remove-Job $inferenceJob
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# ============================================================================
# STEP 6: Open Browser
# ============================================================================
Write-Info "Step 6/6: Opening web interface..."
Start-Sleep -Seconds 2
Start-Process "http://localhost:53801/ui"
Write-Success "Web UI opened in browser"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "   SNN is now running!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Inference UI:  http://localhost:53801/ui" -ForegroundColor Cyan
if ($aimEnabled) {
    Write-Host "  Aim Dashboard: http://localhost:53800" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "  Press Ctrl+C to stop the servers" -ForegroundColor Yellow
Write-Host ""

# Keep script running and monitor the job
try {
    while ($true) {
        if ($inferenceJob.State -ne "Running") {
            Write-Error "Inference server stopped unexpectedly"
            Receive-Job $inferenceJob
            break
        }
        Start-Sleep -Seconds 5
    }
} finally {
    # Cleanup
    Write-Info "Shutting down..."
    Stop-Job $inferenceJob -ErrorAction SilentlyContinue
    Remove-Job $inferenceJob -ErrorAction SilentlyContinue
    
    if ($aimEnabled) {
        Write-Info "Stopping Aim server..."
        Set-Location (Join-Path $ROOT_DIR ".aim_project")
        & docker compose down 2>&1 | Out-Null
        Set-Location $ROOT_DIR
    }
    
    Write-Success "Goodbye!"
}
