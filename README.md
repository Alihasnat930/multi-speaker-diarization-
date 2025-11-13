# 🦷 Dental Voice Intelligence System

**Real-time, self-hosted voice intelligence for dental clinics using 100% open-source models**

A complete, production-ready system for processing dental consultations with:
- ✅ Real-time Voice Activity Detection (VAD) using Silero VAD
- ✅ Speaker diarization (Dentist vs Patient) with SpeechBrain
- ✅ ASR transcription with clinical/dental terminology support
- ✅ Speaker identification via voice embeddings and enrollment
- ✅ Automated SOAP note generation using local LLMs (Mistral/Llama3 with LoRA)

## 🎯 Key Features

### Real-Time Processing
- **Live audio streaming** via WebSocket API
- **Dual microphone support** (separate dentist/patient mics or stereo input)
- **Sub-second latency** for transcription and speaker identification
- **Streaming VAD** with adaptive thresholds

### Clinical Documentation
- **Automated SOAP notes** (Subjective, Objective, Assessment, Plan)
- **Dental terminology optimization** with custom vocabulary
- **Speaker-attributed transcripts** (who said what)
- **Export to EHR-compatible formats**

### Privacy & Compliance
- **100% on-premise** - no cloud dependencies
- **No data leaves your network**
- **HIPAA-ready architecture**
- **Full audit logging**

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or extract the repository
cd speechbrain_dental_engine

# Create virtual environment (Python 3.10+ required)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Enroll Speakers

Before using the system, enroll the dentist and patient voices:

```bash
# Enroll dentist (use a 10-30 second clean audio sample)
python scripts/enroll_speaker.py --id Dentist --audio samples/dentist_voice.wav

# Enroll patient
python scripts/enroll_speaker.py --id Patient --audio samples/patient_voice.wav

# Or use interactive mode
python scripts/enroll_speaker.py --interactive

# List enrolled speakers
python scripts/enroll_speaker.py --list
```

### 3. Start the Service

```bash
# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or use Docker
docker-compose up
```

The API will be available at `http://localhost:8000`

## 📡 Usage

### Real-Time Streaming (Recommended)

Stream live audio from microphone(s) via WebSocket:

```python
# Run the real-time client
python examples/realtime_client.py --mode single

# For dual microphone setup
python examples/realtime_client.py --mode dual --duration 300
```

**WebSocket Protocol:**
```
1. Connect: ws://localhost:8000/ws/stream?mode=single
2. Send: Binary audio chunks (16-bit PCM, 16kHz, mono/stereo)
3. Receive: JSON transcripts with speaker labels
4. Commands: Send JSON {"type": "command", "command": "soap"}
```

### Batch Processing

Process pre-recorded audio files:

```bash
# cURL
curl -X POST "http://localhost:8000/process" \
  -F "file=@consultation_recording.mp3"

# Python
import requests
with open('consultation.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process',
        files={'file': f}
    )
    print(response.json())
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/process` | POST | Batch process audio file |
| `/enroll` | POST | Enroll new speaker |
| `/ws/stream` | WebSocket | Real-time streaming |
| `/sessions/{id}/history` | GET | Get conversation history |
| `/sessions/{id}/soap` | POST | Generate SOAP note |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Audio Input Layer                         │
│  (Microphone/File) → WebSocket/HTTP → Audio Buffer          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Voice Activity Detection (VAD)                  │
│              Silero VAD Model (Real-time)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Diarization Engine                        │
│  Speaker Embeddings (ECAPA-TDNN) + Clustering               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              ASR Transcription Service                       │
│  SpeechBrain ASR + Dental Vocabulary Enhancement            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│             Speaker Identification                           │
│  Voice Embeddings Matching + Enrollment DB                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              SOAP Note Generation                            │
│  Mistral-7B/Llama3 + LoRA (Dental Clinical Notes)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               Output: Transcript + SOAP Note                 │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Model Selection

Edit configuration in the code or use environment variables:

```python
# ASR Model (app/asr_service.py)
model_source = "speechbrain/asr-transformer-transformerlm-librispeech"
# Replace with your fine-tuned model:
# model_source = "./models/asr_dental_finetuned"

# SOAP Generator (app/summarizer_local.py)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# Alternatives:
# model_name = "meta-llama/Llama-3-8B-Instruct"
# model_name = "./models/llama3_dental_lora"

# Speaker Recognition (app/spk_service.py)
source = "speechbrain/spkrec-ecapa-voxceleb"
```

### VAD Parameters

```python
# app/realtime_engine.py
RealtimeVoiceEngine(
    sample_rate=16000,
    vad_threshold=0.5,           # Speech detection threshold (0.0-1.0)
    min_speech_duration=0.3,     # Minimum speech segment (seconds)
    min_silence_duration=0.5,    # Silence before segment end (seconds)
    lookback_duration=30.0       # Buffer size (seconds)
)
```

## 🎓 Training & Fine-Tuning

### Fine-tune ASR for Dental Terminology

```bash
# Prepare dataset (dental conversations with transcripts)
# See scripts/train_asr.py for full example

python scripts/train_asr.py \
  --dataset dental_conversations/ \
  --base_model speechbrain/asr-transformer-transformerlm-librispeech \
  --output models/asr_dental_finetuned \
  --epochs 10
```

### Fine-tune LLM for SOAP Notes

```bash
# Prepare dataset (conversation → SOAP note pairs)
# Use LoRA for efficient fine-tuning

python scripts/train_soap_generator.py \
  --dataset dental_soap_dataset.json \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --output models/dental_soap_lora \
  --lora_r 16 \
  --epochs 3
```

## 📊 Performance

Tested on:
- **CPU**: Intel Core i7 (real-time capable with quantization)
- **GPU**: NVIDIA RTX 3060 (optimal, <500ms latency)
- **RAM**: 16GB minimum, 32GB recommended

Benchmarks:
- **VAD latency**: <50ms
- **ASR latency**: 200-500ms per segment
- **Speaker ID**: <100ms
- **SOAP generation**: 5-15s (depends on conversation length)

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Production deployment with GPU:

```yaml
# docker-compose.yml
services:
  dental-voice-engine:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📝 Example Output

**Transcript:**
```
[0.5s - 3.2s] Dentist: How are you feeling today?
[3.8s - 6.1s] Patient: I've been having some pain in my upper left molar
[6.5s - 12.3s] Dentist: Let me take a look. I can see some decay on tooth number fourteen
[13.0s - 16.8s] Patient: Is it serious?
[17.2s - 25.4s] Dentist: It's a cavity that needs a filling. We can do that today with local anesthesia
```

**SOAP Note:**
```
SUBJECTIVE:
Patient reports pain in upper left molar. No prior dental work in that area.

OBJECTIVE:
Visual examination reveals decay on tooth #14 (upper left first molar).
Moderate caries visible on occlusal surface. No visible abscess.

ASSESSMENT:
Dental caries, tooth #14. Requires restoration.

PLAN:
Proceed with composite filling under local anesthesia (lidocaine with epinephrine).
Follow-up in 2 weeks. Recommend improved oral hygiene and fluoride treatment.
```

## 🔐 Security & Compliance

- All processing happens on-premise
- No external API calls or cloud dependencies
- Audio data never leaves your network
- Speaker embeddings stored locally and encrypted
- Audit logs for all processing activities
- See `docs/compliance.md` for HIPAA considerations

## 🛠️ Troubleshooting

**Issue: Models downloading slowly**
```bash
# Pre-download models
python -c "from app.vad_service import VADService; VADService()"
python -c "from app.asr_service import ASRService; ASRService()"
python -c "from app.spk_service import SpeakerService; SpeakerService()"
```

**Issue: GPU out of memory**
```python
# Enable 4-bit quantization in summarizer_local.py
SOAPNoteGenerator(use_4bit=True)
```

**Issue: Poor transcription accuracy**
- Fine-tune ASR on dental conversations
- Add dental terms to `models/dental_vocabulary.txt`
- Ensure good audio quality (16kHz, low noise)

## 📚 Documentation

- `docs/architecture.md` - System architecture details
- `docs/compliance.md` - HIPAA and privacy considerations
- `docs/training.md` - Model fine-tuning guides
- `docs/api.md` - Complete API documentation

## 🤝 Contributing

This is a self-hosted, open-source solution. Customize as needed for your clinic.

## 📄 License

MIT License - Use freely in your dental practice.

## 🙏 Acknowledgments

Built with:
- [SpeechBrain](https://speechbrain.github.io/) - ASR and speaker recognition
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [Mistral/Llama](https://huggingface.co/models) - SOAP note generation
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

---

**Ready to deploy?** Start with `uvicorn app.main:app --host 0.0.0.0 --port 8000`
