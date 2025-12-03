#!/usr/bin/env python3
"""Launch the ONN web application (frontend + backend).

Usage:
    python launch_web.py           # Start both frontend and backend
    python launch_web.py --backend-only   # Start only the API server
    python launch_web.py --dev     # Start in development mode (hot reload)
"""
import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """Print the ONN banner."""
    banner = f"""
{Colors.CYAN}╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   {Colors.BOLD}OpenNeuralEngine 2.0{Colors.END}{Colors.CYAN}                                      ║
║   {Colors.GREEN}Production-Grade Democratic AI Framework{Colors.CYAN}                  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝{Colors.END}
"""
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed."""
    import importlib.util
    missing = []

    if importlib.util.find_spec("fastapi") is None:
        missing.append("fastapi")

    if importlib.util.find_spec("uvicorn") is None:
        missing.append("uvicorn")

    if missing:
        print(f"{Colors.WARNING}Missing dependencies: {', '.join(missing)}{Colors.END}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True

def check_frontend():
    """Check if frontend is built or node_modules exist."""
    frontend_dir = Path(__file__).parent / "frontend"
    dist_dir = frontend_dir / "dist"
    node_modules = frontend_dir / "node_modules"
    
    return dist_dir.exists() or node_modules.exists()

def start_backend(port: int = 8000, dev: bool = False):
    """Start the FastAPI backend server."""
    import uvicorn

    print(f"{Colors.BLUE}Starting API server on http://localhost:{port}{Colors.END}")

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",  # noqa: S104 # nosec B104 - Intentional for local dev server
        port=port,
        reload=dev,
        log_level="info",
    )

def start_frontend_dev():
    """Start the frontend development server."""
    frontend_dir = Path(__file__).parent / "frontend"

    if not (frontend_dir / "node_modules").exists():
        print(f"{Colors.WARNING}Installing frontend dependencies...{Colors.END}")
        subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=False)  # nosec B603

    print(f"{Colors.BLUE}Starting frontend dev server on http://localhost:3000{Colors.END}")
    subprocess.run(["npm", "run", "dev"], cwd=str(frontend_dir), check=False)  # nosec B603

def build_frontend():
    """Build the frontend for production."""
    frontend_dir = Path(__file__).parent / "frontend"

    if not (frontend_dir / "node_modules").exists():
        print(f"{Colors.WARNING}Installing frontend dependencies...{Colors.END}")
        subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=False)  # nosec B603

    print(f"{Colors.BLUE}Building frontend...{Colors.END}")
    result = subprocess.run(  # nosec B603
        ["npm", "run", "build"], cwd=str(frontend_dir), check=False
    )
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Launch the ONN web application")
    parser.add_argument("--backend-only", action="store_true", help="Start only the API server")
    parser.add_argument("--frontend-only", action="store_true", help="Start only the frontend dev server")
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--build", action="store_true", help="Build frontend before starting")
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Build frontend if requested
    if args.build:
        if not build_frontend():
            print(f"{Colors.FAIL}Frontend build failed{Colors.END}")
            sys.exit(1)
    
    # Start services
    if args.frontend_only:
        start_frontend_dev()
    elif args.backend_only:
        start_backend(args.port, args.dev)
    elif args.dev:
        # Development mode: start both with hot reload
        import threading
        
        # Start backend in thread
        backend_thread = threading.Thread(
            target=start_backend, 
            args=(args.port, True),
            daemon=True
        )
        backend_thread.start()
        
        time.sleep(2)  # Wait for backend to start
        
        # Open browser
        if not args.no_browser:
            webbrowser.open("http://localhost:3000")
        
        # Start frontend (blocking)
        start_frontend_dev()
    else:
        # Production mode: serve built frontend from backend
        frontend_dir = Path(__file__).parent / "frontend"
        dist_dir = frontend_dir / "dist"
        
        if not dist_dir.exists():
            print(f"{Colors.WARNING}Frontend not built. Building now...{Colors.END}")
            if not build_frontend():
                print(f"{Colors.FAIL}Frontend build failed. Running backend only.{Colors.END}")
        
        # Open browser
        if not args.no_browser:
            time.sleep(1)
            webbrowser.open(f"http://localhost:{args.port}")
        
        start_backend(args.port, False)

if __name__ == "__main__":
    main()
