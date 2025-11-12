# Configuration Notes - Lite Version

## Current Configuration: Focus on Speaker ID & Transcription

### ✅ Active Features
- ✅ Real-time audio streaming (WebSocket)
- ✅ Voice Activity Detection (Silero VAD)
- ✅ Speaker diarization and identification
- ✅ Multi-speaker transcription (ASR)
- ✅ Speaker enrollment system
- ✅ Batch audio processing
- ✅ Dual microphone support

### ⏸️ Disabled Features (Commented Out)
- ⏸️ SOAP note generation (LLM)
- ⏸️ Clinical documentation automation
- ⏸️ `/sessions/{id}/soap` endpoint
- ⏸️ Heavy transformers/LLM dependencies

## Why Disabled?

1. **Performance**: LLM models require 4-8GB GPU memory
2. **Startup time**: Faster server initialization
3. **Focus**: Core functionality is speaker ID and transcription
4. **Dependencies**: Reduced package installation time

## Memory Usage

**Before (with LLM):**
- RAM: 16GB recommended
- GPU: 8GB VRAM recommended
- Disk: ~10GB for models

**After (without LLM):**
- RAM: 8GB sufficient
- GPU: 4GB VRAM sufficient (or CPU only)
- Disk: ~2GB for models

## What Still Works

### Batch Processing
```powershell
# Process audio file
curl -X POST "http://localhost:8000/process" -F "file=@consultation.mp3"

# Response includes:
{
  "segments": [
    {"start": 0.5, "end": 3.2, "speaker_id": "Dentist", "text": "...", "score": 0.92},
    {"start": 3.8, "end": 6.1, "speaker_id": "Patient", "text": "...", "score": 0.88}
  ],
  "file_id": "a3b8c9d2",
  "status": "success"
  // Note: 'summary' field removed
}
```

### Real-Time Streaming
```powershell
python examples/realtime_client.py --mode single
```

All speaker identification and transcription features work normally.

### Speaker Enrollment
```powershell
python scripts/enroll_speaker.py --id Dentist --audio sample.wav
```

## To Re-enable LLM/SOAP Features

If you want to re-enable SOAP note generation later:

1. **Uncomment in `app/main.py`:**
   - Import statement: `from app.summarizer_local import summarize_conversation, get_generator`
   - SOAP generation code in `/process` endpoint
   - `/sessions/{id}/soap` endpoint
   - Startup initialization

2. **Uncomment in `app/streaming_api.py`:**
   - `generate_soap_note()` method
   - SOAP command handler in WebSocket

3. **Uncomment in `requirements.txt`:**
   - transformers
   - accelerate
   - bitsandbytes
   - peft
   - Related packages

4. **Reinstall dependencies:**
   ```powershell
   pip install transformers>=4.35.0 accelerate>=0.25.0 bitsandbytes>=0.41.0 peft>=0.7.0
   ```

5. **Restart server**

## API Changes

### Removed Endpoints:
- `POST /sessions/{session_id}/soap` - Generate SOAP note

### Modified Response:
- `POST /process` - No longer includes 'summary' field

### WebSocket Commands:
- `{"command": "soap"}` - No longer supported

## Performance Improvements

**Startup time:**
- Before: 30-60 seconds (loading LLM)
- After: 5-10 seconds

**Memory usage:**
- Before: 8-12GB RAM
- After: 4-6GB RAM

**Processing speed:**
- Same for transcription
- No SOAP generation delay

## Recommended Use Cases

This lite version is perfect for:
- ✅ **Real-time transcription** of consultations
- ✅ **Speaker identification** (who said what)
- ✅ **Conversation logging** with timestamps
- ✅ **Batch processing** of recordings
- ✅ **Integration with other systems** for documentation

For full SOAP note automation, you can:
1. Re-enable LLM features (see above)
2. Use external service for SOAP generation
3. Manual review of transcripts by clinician

## Current Version

**Version:** 1.0.0-lite
**Focus:** Multi-speaker identification and transcription
**Status:** Production-ready for core features

---

**Note:** All documentation files still include LLM features for reference. This configuration is temporary and easily reversible.
