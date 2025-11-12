#!/usr/bin/env python3
"""
Quick verification script to test the lite version setup.
Focuses on speaker identification and transcription features.
"""
import sys
from pathlib import Path

print("\n" + "="*70)
print("  🦷 Dental Voice Intelligence System - Lite Version Check")
print("="*70 + "\n")

print("Checking core components (Speaker ID + Transcription)...\n")

# Check Python version
print("1. Python Version:")
version = sys.version_info
if version.major >= 3 and version.minor >= 10:
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"   ❌ Python {version.major}.{version.minor} (need 3.10+)")
    sys.exit(1)

# Check core imports
print("\n2. Core Dependencies:")
errors = []

try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"      ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"      ℹ️  Running on CPU (GPU not available)")
except ImportError as e:
    print(f"   ❌ PyTorch not found")
    errors.append("torch")

try:
    import speechbrain
    print(f"   ✅ SpeechBrain installed")
except ImportError:
    print(f"   ❌ SpeechBrain not found")
    errors.append("speechbrain")

try:
    import fastapi
    print(f"   ✅ FastAPI {fastapi.__version__}")
except ImportError:
    print(f"   ❌ FastAPI not found")
    errors.append("fastapi")

try:
    import soundfile
    print(f"   ✅ Audio processing (soundfile)")
except ImportError:
    print(f"   ❌ soundfile not found")
    errors.append("soundfile")

try:
    import numpy
    print(f"   ✅ NumPy {numpy.__version__}")
except ImportError:
    print(f"   ❌ NumPy not found")
    errors.append("numpy")

# Check LLM dependencies (should be skipped)
print("\n3. LLM Dependencies (should be commented out):")
try:
    import transformers
    print(f"   ⚠️  Transformers installed (not needed for lite version)")
except ImportError:
    print(f"   ✅ Transformers not installed (correct for lite version)")

try:
    import peft
    print(f"   ⚠️  PEFT installed (not needed for lite version)")
except ImportError:
    print(f"   ✅ PEFT not installed (correct for lite version)")

# Check service imports
print("\n4. Service Components:")
try:
    from app.vad_service import VADService
    print(f"   ✅ VAD Service")
except ImportError as e:
    print(f"   ❌ VAD Service: {e}")
    errors.append("vad_service")

try:
    from app.asr_service import ASRService
    print(f"   ✅ ASR Service")
except ImportError as e:
    print(f"   ❌ ASR Service: {e}")
    errors.append("asr_service")

try:
    from app.spk_service import SpeakerService
    print(f"   ✅ Speaker Recognition Service")
except ImportError as e:
    print(f"   ❌ Speaker Recognition Service: {e}")
    errors.append("spk_service")

try:
    from app.realtime_engine import RealtimeVoiceEngine
    print(f"   ✅ Real-time Engine")
except ImportError as e:
    print(f"   ❌ Real-time Engine: {e}")
    errors.append("realtime_engine")

try:
    from app.streaming_api import stream_manager
    print(f"   ✅ Streaming API")
except ImportError as e:
    print(f"   ❌ Streaming API: {e}")
    errors.append("streaming_api")

# Check directories
print("\n5. Directory Structure:")
dirs_to_check = [
    "models/enrollments",
    "pretrained_models/asr",
    "pretrained_models/spkrec",
    "logs"
]

for dir_path in dirs_to_check:
    if Path(dir_path).exists():
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ⚠️  {dir_path} (will be created on first use)")

# Check enrollment
print("\n6. Speaker Enrollments:")
enroll_dir = Path("models/enrollments")
if enroll_dir.exists():
    enrollments = list(enroll_dir.glob("*.pkl"))
    if enrollments:
        print(f"   ✅ {len(enrollments)} speaker(s) enrolled:")
        for enroll in enrollments:
            print(f"      • {enroll.stem}")
    else:
        print(f"   ℹ️  No speakers enrolled yet")
        print(f"      Run: python scripts/enroll_speaker.py --interactive")
else:
    print(f"   ⚠️  Enrollment directory not found")

# Summary
print("\n" + "="*70)
if errors:
    print("❌ SETUP INCOMPLETE")
    print(f"\nMissing dependencies: {', '.join(errors)}")
    print("\nInstall with: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("✅ SYSTEM READY")
    print("\n🎯 Focus: Multi-Speaker Identification & Transcription")
    print("⏸️  LLM/SOAP features: Disabled")
    
    print("\n📋 Quick Start:")
    print("   1. Enroll speakers:")
    print("      python scripts/enroll_speaker.py --id Dentist --audio sample.wav")
    print("\n   2. Start server:")
    print("      uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("\n   3. Test real-time:")
    print("      python examples/realtime_client.py --mode single")
    print("\n   4. API docs:")
    print("      http://localhost:8000/docs")
    
print("="*70 + "\n")
