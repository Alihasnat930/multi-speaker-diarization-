#!/usr/bin/env python3
"""
Setup and initialization script for Dental Voice Intelligence System.
Handles model downloads, directory creation, and initial configuration.
"""
import os
import sys
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def create_directories():
    """Create necessary directories"""
    logger.info("📁 Creating directory structure...")
    
    directories = [
        'models/enrollments',
        'models/asr_dental_finetuned',
        'models/dental_soap_lora',
        'pretrained_models/asr',
        'pretrained_models/spkrec',
        'pretrained_models/vad',
        'logs',
        'data/temp',
        'data/processed',
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✓ {dir_path}")
    
    logger.info("✅ Directories created\n")


def check_python_version():
    """Verify Python version"""
    logger.info("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error(f"❌ Python 3.10+ required (found {version.major}.{version.minor})")
        return False
    
    logger.info(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install Python dependencies"""
    logger.info("📦 Installing dependencies...")
    logger.info("  This may take several minutes on first run...\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        logger.info("\n✅ Dependencies installed\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Dependency installation failed: {e}")
        return False


def download_models():
    """Pre-download AI models"""
    logger.info("🤖 Downloading AI models (Speaker ID + Transcription focus)...")
    logger.info("  This will download ~1-2GB of models...\n")
    logger.info("  Note: LLM/SOAP models skipped for performance\n")
    
    try:
        # Import and initialize services to trigger model downloads
        logger.info("  • Downloading VAD model...")
        from app.vad_service import VADService
        VADService()
        logger.info("    ✓ VAD model ready")
        
        logger.info("  • Downloading ASR model...")
        from app.asr_service import ASRService
        ASRService()
        logger.info("    ✓ ASR model ready")
        
        logger.info("  • Downloading Speaker Recognition model...")
        from app.spk_service import SpeakerService
        SpeakerService()
        logger.info("    ✓ Speaker Recognition model ready")
        
        logger.info("\n  ℹ️  LLM models skipped (SOAP generation disabled)")
        logger.info("\n✅ Core models downloaded\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        logger.info("\nModels will be downloaded on first use.")
        return False


def verify_gpu():
    """Check GPU availability"""
    logger.info("🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  ✓ GPU available: {gpu_name}")
            logger.info(f"    Memory: {gpu_memory:.1f} GB")
            logger.info("    → Real-time processing will use GPU acceleration")
        else:
            logger.info("  ℹ️  No GPU detected")
            logger.info("    → System will run on CPU (slower but functional)")
    except ImportError:
        logger.info("  ⚠️  PyTorch not installed yet")
    
    print()


def create_sample_config():
    """Create sample configuration file"""
    logger.info("⚙️  Creating configuration file...")
    
    config_content = """# Dental Voice Intelligence System Configuration

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Model Paths
ASR_MODEL=speechbrain/asr-transformer-transformerlm-librispeech
SPEAKER_MODEL=speechbrain/spkrec-ecapa-voxceleb
SOAP_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Processing Settings
SAMPLE_RATE=16000
VAD_THRESHOLD=0.5
MIN_SPEECH_DURATION=0.3
MIN_SILENCE_DURATION=0.5

# Model Options
USE_GPU=auto
USE_4BIT_QUANTIZATION=true
USE_LORA=true
LORA_PATH=models/dental_soap_lora

# Security
ENABLE_CORS=true
LOG_LEVEL=INFO
"""
    
    config_path = Path('.env')
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        logger.info(f"  ✓ Configuration saved to {config_path}")
    else:
        logger.info(f"  ℹ️  Configuration already exists: {config_path}")
    
    print()


def print_next_steps():
    """Print usage instructions"""
    print_header("🎉 Setup Complete!")
    
    print("Next Steps:\n")
    
    print("1️⃣  Enroll Speakers:")
    print("   python scripts/enroll_speaker.py --id Dentist --audio dentist_sample.wav")
    print("   python scripts/enroll_speaker.py --id Patient --audio patient_sample.wav\n")
    
    print("2️⃣  Start the Server:")
    print("   uvicorn app.main:app --host 0.0.0.0 --port 8000\n")
    
    print("3️⃣  Test Real-time Streaming:")
    print("   python examples/realtime_client.py --mode single\n")
    
    print("4️⃣  Access API Documentation:")
    print("   Open http://localhost:8000/docs in your browser\n")
    
    print("=" * 70)
    print("\n📚 Documentation: See README.md and docs/ folder")
    print("🐛 Issues: Check logs/ folder for debugging")
    print("💡 Need help? Review docs/architecture.md\n")


def main():
    """Main setup routine"""
    print_header("🦷 Dental Voice Intelligence System - Setup")
    
    print("This script will:")
    print("  • Check system requirements")
    print("  • Create directory structure")
    print("  • Install dependencies")
    print("  • Download AI models (~2-5GB)")
    print("  • Configure the system\n")
    
    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Setup cancelled.")
        return
    
    # Run setup steps
    if not check_python_version():
        sys.exit(1)
    
    create_directories()
    verify_gpu()
    create_sample_config()
    
    # Ask about dependency installation
    print("📦 Install Python dependencies?")
    print("   (This will run: pip install -r requirements.txt)")
    response = input("   Install? [Y/n]: ").strip().lower()
    
    if not response or response == 'y':
        if not install_dependencies():
            logger.warning("\n⚠️  Some dependencies failed to install.")
            logger.info("You can install them manually with:")
            logger.info("  pip install -r requirements.txt\n")
    
    # Ask about model downloads
    print("\n🤖 Pre-download AI models?")
    print("   (Recommended - saves time on first run)")
    print("   (Requires ~2-5GB download and 5-10 minutes)")
    response = input("   Download now? [Y/n]: ").strip().lower()
    
    if not response or response == 'y':
        download_models()
    else:
        logger.info("  ℹ️  Models will be downloaded automatically on first use\n")
    
    print_next_steps()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        sys.exit(1)
