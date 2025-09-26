#!/usr/bin/env python3
"""
VEX Tournament Analyzer Launcher
Simple script to launch the Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Launch the VEX Tournament Analyzer application."""
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the VEX Tournament Analyzer directory")
        sys.exit(1)
    
    # Check if Excel file exists
    if not os.path.exists('GreatPlanesTeamData.xlsx'):
        print("⚠️  Warning: GreatPlanesTeamData.xlsx not found")
        print("Please ensure your Excel file is in the same directory")
    
    print("🚀 Starting VEX Tournament Analyzer...")
    print("📊 Loading team data and initializing application...")
    print("🌐 The application will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        # Launch Streamlit app
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
