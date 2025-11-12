"""
Simple server starter script that doesn't fork processes.
Run this directly with: python run_server_simple.py
"""
import uvicorn
import os

if __name__ == "__main__":
    # Disable symlinks on Windows to avoid privilege issues
    os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run server without reload to avoid multiprocessing issues on Windows
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
