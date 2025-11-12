# Quick Start Guide

Get the Dental Voice Intelligence System running in 5 minutes!

## Prerequisites

- **Python 3.10+** installed
- **8GB RAM** minimum (16GB+ recommended)
- **GPU optional** (NVIDIA GPU recommended for best performance)
- **Microphone** for real-time testing

## Step 1: Install Dependencies

```powershell
# Navigate to project directory
cd speechbrain_dental_engine

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**This will take 5-10 minutes** to download all packages.

## Step 2: Run Setup Script

```powershell
# Run automated setup
python setup.py
```

This script will:
- ✅ Create necessary directories
- ✅ Check Python version
- ✅ Download AI models (~2-5GB)
- ✅ Create configuration file

**Expected time: 5-15 minutes** (depending on internet speed)

## Step 3: Enroll Speakers

Before the system can identify speakers, you need to enroll them:

### Option A: Interactive Mode

```powershell
python scripts/enroll_speaker.py --interactive
```

Follow the prompts to enroll speakers.

### Option B: Command Line

```powershell
# Enroll dentist
python scripts/enroll_speaker.py --id Dentist --audio path\to\dentist_sample.wav

# Enroll patient
python scripts/enroll_speaker.py --id Patient --audio path\to\patient_sample.wav
```

**Requirements for enrollment audio:**
- Duration: 10-30 seconds of clean speech
- Format: WAV, MP3, or M4A
- Quality: Clear voice, minimal background noise

**Don't have sample audio?** You can skip this step and speakers will be labeled as "Unknown" until enrolled.

## Step 4: Start the Server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Server is ready!** 🎉

## Step 5: Test the System

### Option A: Web Interface

Open your browser and go to:
```
http://localhost:8000/docs
```

You'll see the interactive API documentation where you can:
- Upload audio files
- View responses
- Test all endpoints

### Option B: Batch Processing Example

```powershell
# Process a pre-recorded consultation
python examples/batch_demo.py --file path\to\consultation.mp3
```

### Option C: Real-Time Streaming

**Install audio dependencies first:**
```powershell
pip install pyaudio websockets
```

**Then run the streaming client:**
```powershell
python examples/realtime_client.py --mode single --duration 60
```

This will:
1. Connect to the server
2. Start recording from your microphone
3. Stream audio in real-time
4. Display transcripts as they're generated
5. Stop after 60 seconds (or press Ctrl+C)

## Example Output

### Transcript:
```
[    0.5s -     3.2s]    Dentist: How are you feeling today?
[    3.8s -     6.1s]    Patient: I've been having some pain in my upper left molar
[    6.5s -    12.3s]    Dentist: Let me take a look. I can see some decay on tooth number fourteen
[   13.0s -    16.8s]    Patient: Is it serious?
[   17.2s -    25.4s]    Dentist: It's a cavity that needs a filling
```

### SOAP Note:
```
SOAP Clinical Note
==================

SUBJECTIVE:
Patient reports pain in upper left molar for 3 days.

OBJECTIVE:
Visual examination reveals decay on tooth #14 (upper left first molar).
Moderate caries visible on occlusal surface.

ASSESSMENT:
Dental caries, tooth #14. Requires restoration.

PLAN:
Composite filling under local anesthesia.
Follow-up in 2 weeks.
```

## Common Issues

### Issue: "Module not found" errors

**Solution:**
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "CUDA not available" warning

**This is OK!** The system will run on CPU (slower but functional).

To enable GPU:
1. Install NVIDIA GPU drivers
2. Install CUDA Toolkit 11.8+
3. Reinstall PyTorch with CUDA support:
   ```powershell
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: Server starts but models not downloading

**Solution:** Models download on first use. Wait for first request or pre-download:
```powershell
python -c "from app.vad_service import VADService; VADService()"
python -c "from app.asr_service import ASRService; ASRService()"
python -c "from app.spk_service import SpeakerService; SpeakerService()"
```

### Issue: Poor transcription accuracy

**Solutions:**
1. Ensure good audio quality (16kHz, low noise)
2. Enroll speakers properly
3. Add dental terms to `models/dental_vocabulary.txt`
4. Consider fine-tuning ASR model (see `docs/training.md`)

### Issue: PyAudio installation fails

**On Windows:**
```powershell
pip install pipwin
pipwin install pyaudio
```

**On Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### Issue: WebSocket connection refused

**Check:**
1. Server is running: `http://localhost:8000/health`
2. No firewall blocking port 8000
3. Using correct URL: `ws://localhost:8000/ws/stream`

## Next Steps

### 1. Test with Real Conversations

Record or use existing dental consultation recordings and process them:

```powershell
python examples/batch_demo.py --directory path\to\recordings
```

### 2. Fine-tune Models (Optional)

For better accuracy on your clinic's specific terminology and speakers:

See `docs/training.md` for detailed instructions.

### 3. Deploy for Production

**Option A: Run as Windows Service**

Use NSSM or Task Scheduler to run the server automatically on boot.

**Option B: Docker Deployment**

```powershell
docker-compose up -d
```

**Option C: Cloud Deployment**

Deploy on your own server (not recommended for PHI - use on-premise).

### 4. Integration with EHR

The API can be integrated with your existing EHR system:

```python
# Example integration
import requests

# Process consultation
response = requests.post(
    'http://localhost:8000/process',
    files={'file': open('consultation.wav', 'rb')}
)

soap_note = response.json()['summary']

# Save to EHR
ehr_api.create_note(patient_id, soap_note)
```

## Performance Tips

1. **Use GPU** - 10x faster processing
2. **Optimize audio** - 16kHz sample rate, remove silence
3. **Enroll speakers** - Better accuracy when speakers are enrolled
4. **Fine-tune models** - Train on your clinic's data for best results
5. **Batch processing** - Process multiple files overnight

## Getting Help

- **Documentation**: See `README.md` and files in `docs/` folder
- **API Reference**: `docs/api.md`
- **Training Guide**: `docs/training.md`
- **Architecture**: `docs/architecture.md`

## What's Next?

- ✅ System is running
- ✅ You can process audio files
- ✅ You can stream real-time audio

**You're ready to use the Dental Voice Intelligence System!**

For production deployment, review:
- `docs/compliance.md` - HIPAA considerations
- `docs/architecture.md` - System design
- Security best practices (encryption, authentication, audit logs)

---

**Need help?** Check the documentation or review example files in `examples/` folder.
