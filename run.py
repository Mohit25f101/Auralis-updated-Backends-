#!/usr/bin/env python
# ==============================
# 📄 run.py
# ==============================
# Simple script to run Auralis
# ==============================

import subprocess
import sys
import os

def main():
    """Run the Auralis server."""
    print("\n" + "="*60)
    print("🚀 AURALIS ML SYSTEM LAUNCHER")
    print("="*60 + "\n")
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found!")
        print("   Make sure you're in the auralis directory.")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required!")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print("📦 Checking dependencies...")
    
    # Try to import required packages
    required = ['fastapi', 'uvicorn', 'tensorflow', 'librosa', 'transformers']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        
        response = input("\nInstall now? (y/n): ").strip().lower()
        if response == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            sys.exit(1)
    
    print("\n" + "="*60)
    print("🌐 Starting Auralis Server...")
    print("="*60)
    print("\n   URL: http://127.0.0.1:8000")
    print("   Docs: http://127.0.0.1:8000/docs")
    print("\n   Press Ctrl+C to stop\n")
    
    # Run the server
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped.")


if __name__ == "__main__":
    main()