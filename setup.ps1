# ==============================
# 📄 setup.ps1 - Auralis Ultimate Setup Script
# ==============================
# Run this script in PowerShell to create the complete project structure
# Usage: .\setup.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "🚀 AURALIS ULTIMATE  - Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) {
    $ScriptDir = Get-Location
}

Write-Host "📁 Project directory: $ScriptDir" -ForegroundColor Yellow
Write-Host ""

# Create directory structure
Write-Host "📂 Creating directory structure..." -ForegroundColor Green

$directories = @(
    "api",
    "api\routes",
    "api\models",
    "api\middleware",
    "services",
    "utils",
    "data",
    "tests",
    "logs"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $ScriptDir $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "   ✅ Created: $dir" -ForegroundColor Gray
    } else {
        Write-Host "   ⏭️  Exists: $dir" -ForegroundColor DarkGray
    }
}

Write-Host ""

# Create __init__.py files
Write-Host "📝 Creating __init__.py files..." -ForegroundColor Green

$initFiles = @(
    "api\__init__.py",
    "api\routes\__init__.py",
    "api\models\__init__.py",
    "api\middleware\__init__.py",
    "services\__init__.py",
    "utils\__init__.py"
)

foreach ($initFile in $initFiles) {
    $fullPath = Join-Path $ScriptDir $initFile
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType File -Path $fullPath -Force | Out-Null
        Write-Host "   ✅ Created: $initFile" -ForegroundColor Gray
    }
}

Write-Host ""

# Create .env file
Write-Host "📝 Creating .env file..." -ForegroundColor Green

$envContent = @"
# Auralis Ultimate Environment Configuration

# Server Settings
HOST=127.0.0.1
PORT=8000
DEBUG=false

# Model Settings
WHISPER_MODEL=openai/whisper-small

# FFmpeg Path (update this for your system)
FFMPEG_PATH=D:\photo\ffmpeg\ffmpeg-2026-01-07-git-af6a1dd0b2-full_build\bin

# Data Settings
DATA_DIR=data
"@

$envPath = Join-Path $ScriptDir ".env"
if (-not (Test-Path $envPath)) {
    Set-Content -Path $envPath -Value $envContent
    Write-Host "   ✅ Created: .env" -ForegroundColor Gray
}

Write-Host ""

# Create .gitignore
Write-Host "📝 Creating .gitignore..." -ForegroundColor Green

$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Data
data/*.json
logs/*.log

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
"@

$gitignorePath = Join-Path $ScriptDir ".gitignore"
if (-not (Test-Path $gitignorePath)) {
    Set-Content -Path $gitignorePath -Value $gitignoreContent
    Write-Host "   ✅ Created: .gitignore" -ForegroundColor Gray
}

Write-Host ""

# Summary
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 Project Structure Created:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   auralis_ultimate/"
Write-Host "   ├── api/"
Write-Host "   │   ├── routes/"
Write-Host "   │   │   ├── analyze.py"
Write-Host "   │   │   ├── health.py"
Write-Host "   │   │   ├── feedback.py"
Write-Host "   │   │   └── auth.py"
Write-Host "   │   ├── models/"
Write-Host "   │   │   ├── requests.py"
Write-Host "   │   │   └── responses.py"
Write-Host "   │   └── middleware/"
Write-Host "   │       └── error_handler.py"
Write-Host "   ├── services/"
Write-Host "   │   ├── audio_loader.py"
Write-Host "   │   ├── whisper_manager.py"
Write-Host "   │   ├── yamnet_manager.py"
Write-Host "   │   ├── emotion_detector.py"
Write-Host "   │   ├── confidence_scorer.py"
Write-Host "   │   ├── context_synthesizer.py"
Write-Host "   │   ├── learning_system.py"
Write-Host "   │   └── analyzer.py"
Write-Host "   ├── utils/"
Write-Host "   │   ├── audio_utils.py"
Write-Host "   │   ├── text_utils.py"
Write-Host "   │   └── validation.py"
Write-Host "   ├── data/"
Write-Host "   ├── logs/"
Write-Host "   ├── config.py"
Write-Host "   ├── main.py"
Write-Host "   ├── requirements.txt"
Write-Host "   └── .env"
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   1. Create virtual environment:"
Write-Host "      python -m venv venv" -ForegroundColor Cyan
Write-Host ""
Write-Host "   2. Activate virtual environment:"
Write-Host "      .\venv\Scripts\Activate" -ForegroundColor Cyan
Write-Host ""
Write-Host "   3. Install dependencies:"
Write-Host "      pip install -r requirements.txt" -ForegroundColor Cyan
Write-Host ""
Write-Host "   4. Copy all Python files to their locations"
Write-Host ""
Write-Host "   5. Update .env with your FFmpeg path"
Write-Host ""
Write-Host "   6. Run the application:"
Write-Host "      python main.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "   7. Open in browser:"
Write-Host "      http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan