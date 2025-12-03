# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-03

### 🎉 Major Release - Production Ready!

This release transforms SNN into a production-ready application with one-click deployment, like Spotify or Unity Hub.

### Added

#### One-Click Launcher
- **Start-SNN.bat** - Windows batch file for double-click launching
- **Start-SNN.ps1** - PowerShell automation script with:
  - Automatic Python version detection (3.11+ required)
  - Docker availability checking
  - Virtual environment creation and activation
  - Dependency installation
  - Aim server startup via Docker Compose
  - Inference server launch with health checks
  - Automatic browser opening to UI
  - Graceful degradation when Docker unavailable
  - Colorized progress output and error reporting

#### Docker & Deployment
- **.aim_project/docker-compose.yml** - Aim tracking server configuration
  - Persistent volume for experiment data
  - Health checks for container orchestration
  - Automatic restart policy
- **Dockerfile** - Production-ready container image
  - Multi-stage build optimization
  - Health check integration
  - Minimal attack surface
- **.dockerignore** - Optimized Docker build context
- **DEPLOYMENT.md** - Comprehensive deployment guide:
  - Local one-click setup
  - Docker deployment recipes
  - Kubernetes manifests
  - Production best practices
  - Monitoring and health checks
  - Backup and recovery procedures

#### Testing & CI/CD
- **pytest.ini** - Fixed pytest configuration with:
  - Proper coverage settings
  - Test markers for categorization
  - Excluded directories
  - HTML and terminal coverage reports
- **.github/workflows/tests.yml** - Automated testing:
  - Runs on Ubuntu and Windows
  - Python 3.11 and 3.12 support
  - Automatic coverage upload
  - Triggered on push to main/develop and PRs

#### Configuration
- **.env** - Environment variables with sensible defaults:
  - Aim tracking URI
  - Tracking mode (enabled/disabled)
  - Log levels
  - Python path configuration
- Auto-configuration for Aim connection
- Fallback modes when services unavailable

#### Documentation
- **QUICKSTART.md** - 60-second getting started guide
- Updated **README.md** with one-click instructions
- **.github/copilot-instructions.md** - AI coding agent guidelines
- Enhanced inline documentation throughout

### Changed

#### Inference Server
- Added `/ready` endpoint for deployment orchestration
- Enhanced `/health` endpoint with timestamp and detailed status
- Improved error handling and user feedback
- Better model loading error messages

#### Console Scripts
- Fixed `pyproject.toml` entry points to use correct module paths:
  - `snn-evaluate` → `scripts.eval_model:main`
  - `snn-inference` → `scripts.launch_aim_inference:main`

#### User Experience
- **Zero terminal commands required** for end users
- Automatic prerequisite checking
- Visual progress indicators
- Friendly error messages with solutions
- Automatic browser launching
- Self-diagnosing health checks

### Fixed

- Coverage tests now work correctly with pytest.ini
- Console script module path resolution
- Docker Compose missing from repository
- Aim server connection handling
- Health endpoint compatibility for load balancers

### Infrastructure

- Production-ready Docker setup
- CI/CD pipeline with automated tests
- Multi-platform support (Windows, Linux, macOS)
- Scalable deployment options (Docker, Kubernetes)

---

## [0.1.0] - Previous Version

Initial release with:
- PuzzleModel architecture
- Text and multimodal support
- Aim experiment tracking
- Training, evaluation, and inference capabilities
- HuggingFace model integration

---

## Upgrade Guide

### From 0.1.0 to 1.0.0

**For End Users:**
1. Pull the latest code
2. Double-click `Start-SNN.bat` - that's it!

**For Developers:**
1. Pull the latest code
2. Run: `pip install -e .` to update entry points
3. Use `Start-SNN.bat` or continue with manual commands

**For Production:**
1. Build new Docker image: `docker build -t snn-inference:1.0.0 .`
2. Update Kubernetes manifests with new health check endpoints
3. Review DEPLOYMENT.md for production best practices

**Breaking Changes:**
- Console script paths updated (but backwards compatible)
- New environment variables in `.env` (optional)
- Health check endpoints added (new feature)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
