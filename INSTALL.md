# 🦷 Complete Installation & Setup Guide

## What You're Installing

A **production-ready, real-time voice intelligence system** for dental clinics featuring:
- 🎙️ Live audio streaming with dual microphone support
- 🗣️ Speaker diarization and identification (Dentist vs Patient)
- 📝 Real-time ASR transcription with dental terminology
- 📋 Automated SOAP note generation using local LLM
- 🔒 100% on-premise (no cloud dependencies)

---

## Installation Methods

Choose the method that works best for you:

### Method 1: Automated Setup (Recommended)
### Method 2: Manual Installation
### Method 3: Docker Deployment

---

## Method 1: Automated Setup ⚡ (5-15 minutes)

**Best for:** Quick start, development, testing

### Step 1: Prerequisites

```powershell
# Check Python version (must be 3.10+)
python --version

# If Python not installed, download from:
# https://www.python.org/downloads/
```

### Step 2: Run Setup Script

```powershell
# Navigate to project folder
cd speechbrain_dental_engine

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Run automated setup
python setup.py
```

The setup script will:
- ✅ Check system requirements
- ✅ Create directory structure
- ✅ Install dependencies (~5 minutes)
- ✅ Download AI models (~5-10 minutes, 2-5GB)
- ✅ Create configuration file

### Step 3: Enroll Speakers

```powershell
# Interactive enrollment
python scripts/enroll_speaker.py --interactive

# Or command line
python scripts/enroll_speaker.py --id Dentist --audio dentist_sample.wav
python scripts/enroll_speaker.py --id Patient --audio patient_sample.wav
```

### Step 4: Start Server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Done!** Open http://localhost:8000/docs to see the API.

---

## Method 2: Manual Installation (15-30 minutes)

**Best for:** Custom configuration, understanding the system

### Step 1: Install Python 3.10+

Download from: https://www.python.org/downloads/

During installation:
- ☑️ Check "Add Python to PATH"
- ☑️ Install pip

### Step 2: Create Project Structure

```powershell
cd speechbrain_dental_engine

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install main dependencies
pip install -r requirements.txt

# This takes 5-10 minutes
```

**If you encounter errors:**

```powershell
# For PyTorch with CUDA (GPU support)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For PyAudio (if it fails)
pip install pipwin
pipwin install pyaudio
```

### Step 4: Create Directories

```powershell
# Create model directories
New-Item -ItemType Directory -Force -Path models\enrollments
New-Item -ItemType Directory -Force -Path pretrained_models\asr
New-Item -ItemType Directory -Force -Path pretrained_models\spkrec
New-Item -ItemType Directory -Force -Path logs
```

### Step 5: Download Models (Optional)

Pre-download models to save time later:

```powershell
# This downloads ~2-5GB
python -c "from app.vad_service import VADService; VADService()"
python -c "from app.asr_service import ASRService; ASRService()"
python -c "from app.spk_service import SpeakerService; SpeakerService()"
```

### Step 6: Configure (Optional)

Create `.env` file:

```env
API_HOST=0.0.0.0
API_PORT=8000
USE_GPU=auto
USE_4BIT_QUANTIZATION=true
LOG_LEVEL=INFO
```

### Step 7: Enroll Speakers

```powershell
python scripts/enroll_speaker.py --id Dentist --audio dentist_sample.wav
python scripts/enroll_speaker.py --id Patient --audio patient_sample.wav
```

### Step 8: Start Server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Method 3: Docker Deployment 🐳 (10-20 minutes)

**Best for:** Production, consistent environments

### Prerequisites

1. **Install Docker Desktop** (Windows/Mac)
   - Download: https://www.docker.com/products/docker-desktop

2. **For GPU Support** (Optional but recommended)
   - Install NVIDIA drivers
   - Install NVIDIA Container Toolkit
   - See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Step 1: Build and Run

```powershell
cd speechbrain_dental_engine

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f
```

### Step 2: Wait for Startup

First startup takes 5-10 minutes as models download.

Check status:
```powershell
# Check if container is running
docker-compose ps

# Check health
curl http://localhost:8000/health
```

### Step 3: Enroll Speakers

```powershell
# Access container
docker-compose exec dental-voice-engine bash

# Inside container
python scripts/enroll_speaker.py --interactive
```

### Step 4: Access API

Open http://localhost:8000/docs

---

## Verification & Testing

### 1. Check Installation

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Test imports
python -c "import speechbrain; import torch; import fastapi; print('✅ All imports successful')"

# Check GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

### 2. Test API

```powershell
# Health check
curl http://localhost:8000/health

# Should return: {"status":"healthy","service":"dental-voice-engine"}
```

### 3. Test Batch Processing

```powershell
# Process a sample audio file
python examples/batch_demo.py --file path\to\audio.wav
```

### 4. Test Real-Time Streaming

```powershell
# Install client dependencies
pip install pyaudio websockets

# Run streaming client
python examples/realtime_client.py --mode single --duration 30
```

---

## Troubleshooting

### Issue: Python not found

**Solution:**
```powershell
# Download and install Python 3.10+ from python.org
# Make sure to check "Add Python to PATH" during installation
```

### Issue: pip install fails

**Solution:**
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Try installing with --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

### Issue: CUDA/GPU not detected

**This is OK!** The system works on CPU (just slower).

**To enable GPU:**
1. Install NVIDIA drivers from: https://www.nvidia.com/drivers
2. Install CUDA Toolkit 11.8+: https://developer.nvidia.com/cuda-downloads
3. Reinstall PyTorch with CUDA:
   ```powershell
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: Models not downloading

**Solution:**
```powershell
# Check internet connection
# Models download on first use from HuggingFace
# Requires ~2-5GB download

# Manually trigger download
python -c "from app.vad_service import VADService; VADService()"
```

### Issue: Port 8000 already in use

**Solution:**
```powershell
# Use different port
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Or find and kill process using port 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

### Issue: PyAudio installation fails

**Windows Solution:**
```powershell
pip install pipwin
pipwin install pyaudio
```

**Linux Solution:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### Issue: Module 'speechbrain' not found

**Solution:**
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall speechbrain
pip install speechbrain>=0.5.12
```

### Issue: Out of memory errors

**Solutions:**
1. Enable 4-bit quantization (edit `app/summarizer_local.py`, set `use_4bit=True`)
2. Use smaller LLM model
3. Close other applications
4. Add more RAM or use GPU

---

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 15GB free space
- **OS**: Windows 10+, Ubuntu 20.04+, macOS 10.15+
- **Python**: 3.10 or higher

### Recommended Requirements
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7)
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060, RTX 4060, or better)
- **Storage**: 20GB+ SSD
- **Network**: 100Mbps+ for model downloads

### For Production
- **CPU**: 16+ cores
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070, A4000, or better)
- **Storage**: 50GB+ SSD with RAID
- **Network**: Dedicated server on clinic network

---

## Post-Installation

### 1. Configuration

Edit `.env` file or environment variables:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
USE_GPU=auto                    # auto, cuda, cpu
USE_4BIT_QUANTIZATION=true      # Reduce memory usage
USE_LORA=true                   # Use fine-tuned adapters

# Processing Settings
VAD_THRESHOLD=0.5               # Speech detection sensitivity
MIN_SPEECH_DURATION=0.3         # Minimum speech segment (seconds)
MIN_SILENCE_DURATION=0.5        # Silence before segment end

# Logging
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
```

### 2. Security Setup

**For production deployment:**

```python
# Add to app/main.py
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/process")
async def process(
    file: UploadFile,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify credentials
    # ... existing code
```

### 3. Integration with EHR

Example integration:

```python
import requests

# Process consultation
response = requests.post(
    'http://localhost:8000/process',
    files={'file': open('consultation.wav', 'rb')}
)

result = response.json()
soap_note = result['summary']

# Save to your EHR system
ehr_api.create_clinical_note(
    patient_id=patient_id,
    note_text=soap_note,
    note_type='SOAP'
)
```

---

## Quick Reference

### Start Server
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Process Audio File
```powershell
python examples/batch_demo.py --file consultation.wav
```

### Real-Time Streaming
```powershell
python examples/realtime_client.py --mode single
```

### Enroll Speaker
```powershell
python scripts/enroll_speaker.py --id Dentist --audio sample.wav
```

### View API Docs
```
http://localhost:8000/docs
```

### Check Logs
```powershell
Get-Content logs\app.log -Tail 50 -Wait
```

---

## Next Steps

After installation:

1. ✅ **Test with sample audio** - Verify system works
2. ✅ **Enroll clinic staff** - Better speaker identification
3. ✅ **Review accuracy** - Test with real conversations
4. ✅ **Fine-tune models** - Improve for your clinic (optional)
5. ✅ **Deploy** - Move to production server
6. ✅ **Train staff** - Show how to use system
7. ✅ **Monitor** - Track performance and accuracy

---

## Getting Help

- 📚 **Documentation**: See `README.md` and files in `docs/`
- 🔧 **API Reference**: `docs/api.md`
- 🎓 **Training Guide**: `docs/training.md`
- 🏗️ **Architecture**: `docs/architecture.md`
- 🚀 **Quick Start**: `QUICKSTART.md`

---

## Success Checklist

Before going live:

- [ ] Python 3.10+ installed
- [ ] All dependencies installed (`pip list` shows speechbrain, torch, fastapi, etc.)
- [ ] Server starts without errors
- [ ] Health check returns OK: `curl http://localhost:8000/health`
- [ ] Can process test audio file
- [ ] Speakers enrolled (Dentist, Patient, etc.)
- [ ] Real-time streaming works
- [ ] SOAP notes generate correctly
- [ ] Reviewed security considerations
- [ ] Backup strategy in place
- [ ] Staff trained on usage

---

**You're ready to go!** 🎉

For production deployment, review the compliance documentation in `docs/compliance.md` and implement appropriate security measures.
