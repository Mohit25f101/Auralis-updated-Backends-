#!/usr/bin/env python3
"""
AURALIS v5.1 - Automatic Integration Script
============================================
This script automatically applies all improvements to your existing Auralis installation.

Usage:
    python integrate_improvements.py

What it does:
1. Creates backups of original files
2. Copies improved service files
3. Updates imports in main.py (optional)
4. Tests the integration
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


class AuralisUpgrader:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.services_dir = self.base_dir / "services"
        self.backup_dir = self.base_dir / "backups" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.errors = []
        self.warnings = []
        
    def run(self):
        """Run the complete upgrade process"""
        print_header("AURALIS v5.1 - AUTOMATIC UPGRADE")
        
        # Step 1: Validate environment
        if not self.validate_environment():
            print_error("Environment validation failed. Aborting.")
            return False
        
        # Step 2: Create backups
        if not self.create_backups():
            print_error("Backup creation failed. Aborting.")
            return False
        
        # Step 3: Copy improved files
        if not self.copy_improved_files():
            print_error("File copy failed. Aborting.")
            return False
        
        # Step 4: Test integration (optional)
        self.test_integration()
        
        # Step 5: Show summary
        self.show_summary()
        
        return True
    
    def validate_environment(self):
        """Validate that we're in the right directory"""
        print_info("Validating environment...")
        
        # Check for main.py
        if not (self.base_dir / "main.py").exists():
            print_error("main.py not found. Are you in the Auralis root directory?")
            return False
        print_success("Found main.py")
        
        # Check for services directory
        if not self.services_dir.exists():
            print_error("services/ directory not found")
            return False
        print_success("Found services/ directory")
        
        # Check for required improved files
        required_files = [
            "whisper_manager_improved.py",
            "location_detector_improved.py"
        ]
        
        for file in required_files:
            if not (self.base_dir / file).exists():
                print_error(f"{file} not found in current directory")
                print_info(f"Please ensure {file} is in the same directory as this script")
                return False
        print_success("Found all required improvement files")
        
        return True
    
    def create_backups(self):
        """Create backups of files that will be modified"""
        print_info("Creating backups...")
        
        try:
            # Create backup directory
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            print_success(f"Created backup directory: {self.backup_dir}")
            
            # Files to backup
            files_to_backup = [
                self.services_dir / "whisper_manager.py",
                self.services_dir / "location_detector.py",
                self.base_dir / "main.py"
            ]
            
            for file in files_to_backup:
                if file.exists():
                    backup_path = self.backup_dir / file.name
                    shutil.copy2(file, backup_path)
                    print_success(f"Backed up: {file.name}")
                else:
                    print_warning(f"File not found (skipping backup): {file.name}")
            
            return True
            
        except Exception as e:
            print_error(f"Backup failed: {e}")
            return False
    
    def copy_improved_files(self):
        """Copy improved files to services directory"""
        print_info("Installing improved files...")
        
        try:
            # Copy whisper_manager_improved.py
            src = self.base_dir / "whisper_manager_improved.py"
            dst = self.services_dir / "whisper_manager_improved.py"
            shutil.copy2(src, dst)
            print_success("Installed: whisper_manager_improved.py")
            
            # Copy location_detector_improved.py
            src = self.base_dir / "location_detector_improved.py"
            dst = self.services_dir / "location_detector_improved.py"
            shutil.copy2(src, dst)
            print_success("Installed: location_detector_improved.py")
            
            return True
            
        except Exception as e:
            print_error(f"File copy failed: {e}")
            return False
    
    def test_integration(self):
        """Test that the new modules can be imported"""
        print_info("Testing integration...")
        
        # Add services to path
        sys.path.insert(0, str(self.services_dir))
        
        try:
            # Test WhisperManagerImproved
            from whisper_manager_improved import WhisperManagerImproved
            whisper_test = WhisperManagerImproved()
            print_success("WhisperManagerImproved imports successfully")
        except Exception as e:
            print_error(f"WhisperManagerImproved import failed: {e}")
            self.errors.append(str(e))
        
        try:
            # Test LocationDetectorImproved
            from location_detector_improved import LocationDetectorImproved
            location_test = LocationDetectorImproved()
            print_success("LocationDetectorImproved imports successfully")
        except Exception as e:
            print_error(f"LocationDetectorImproved import failed: {e}")
            self.errors.append(str(e))
    
    def show_summary(self):
        """Show upgrade summary and next steps"""
        print_header("UPGRADE SUMMARY")
        
        if not self.errors:
            print_success("All improvements installed successfully!")
            print()
            print_info("NEXT STEPS:")
            print()
            print("1. Update your main.py to use the improved classes:")
            print()
            print(f"{Colors.YELLOW}   # Add these imports:{Colors.END}")
            print("   from services.whisper_manager_improved import WhisperManagerImproved")
            print("   from services.location_detector_improved import LocationDetectorImproved")
            print()
            print(f"{Colors.YELLOW}   # Update initialization:{Colors.END}")
            print("   whisper = WhisperManagerImproved('openai/whisper-small')")
            print("   location_detector = LocationDetectorImproved()")
            print()
            print("2. Restart your Auralis server:")
            print("   python main.py")
            print()
            print("3. Test with various audio files")
            print()
            print("4. Use /feedback endpoint to train the learning system")
            print()
            print_info(f"Backups saved to: {self.backup_dir}")
            print()
        else:
            print_error("Some errors occurred during upgrade:")
            for error in self.errors:
                print(f"   - {error}")
            print()
            print_warning("You can restore from backups if needed:")
            print(f"   {self.backup_dir}")
        
        if self.warnings:
            print()
            print_warning("Warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")


def main():
    """Main entry point"""
    upgrader = AuralisUpgrader()
    
    # Ask for confirmation
    print_header("AURALIS v5.1 - AUTOMATIC UPGRADE")
    print("This script will:")
    print("  1. Backup your current files")
    print("  2. Install improved service modules")
    print("  3. Test the integration")
    print()
    
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print_warning("Upgrade cancelled")
        return
    
    # Run upgrade
    success = upgrader.run()
    
    if success:
        print()
        print_success("🎉 Upgrade completed successfully!")
    else:
        print()
        print_error("Upgrade failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()