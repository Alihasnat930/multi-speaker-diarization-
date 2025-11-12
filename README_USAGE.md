# Multi-Speaker Identification System - Setup Complete! 🎉

## ✅ System Status: WORKING

Your dental voice intelligence system with multi-speaker identification is configured and ready!

## 🎯 What It Does

- **Detects multiple speakers** in audio conversations
- **Separates speech** by different people with timestamps
- **Transcribes each speaker** separately
- **Labels speakers** (Speaker_0, Speaker_1, etc.)
- **Dental terminology** support built-in

## 🚀 How to Use

### 1. Start the Server
```powershell
cd "c:\Users\hjiaz tr\Downloads\speechbrain_dental_engine"
.\START_SERVER.bat
```
Or double-click `START_SERVER.bat`

### 2. Upload Audio
- Open: http://localhost:8000
- Drag & drop your audio file (WAV, MP3, M4A)
- Wait for processing
- View results with speaker segments!

### 3. Expected Output
```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker_id": "Speaker_0",
      "score": 0.85,
      "text": "Hello, how can I help you today?"
    },
    {
      "start": 3.5,
      "end": 6.8,
      "speaker_id": "Speaker_1",
      "score": 0.92,
      "text": "I have pain in my upper molar"
    }
  ]
}
```

## ⚠️ Windows Symlink Issue (RESOLVED)

### Problem
Windows requires admin privileges for symlinks. HuggingFace Hub tries to create symlinks when downloading models.

### Solutions Implemented
1. ✅ Environment variable: `HF_HUB_DISABLE_SYMLINKS=1`
2. ✅ Automatic file copying when symlink errors occur
3. ✅ Models pre-downloaded to `pretrained_models/spkrec/`
4. ✅ Graceful error handling

### If You Still See Symlink Errors
The models are already downloaded. The system will copy files automatically on first use.

## 📁 Project Structure

```
speechbrain_dental_engine/
├── app/
│   ├── main.py              # FastAPI server
│   ├── diarization.py       # Multi-speaker processing
│   ├── spk_service.py       # Speaker identification
│   ├── asr_service.py       # Speech-to-text
│   └── vad_service.py       # Voice activity detection
├── pretrained_models/
│   └── spkrec/              # Downloaded models (83MB+)
├── models/
│   ├── dental_vocabulary.txt
│   └── enrollments/         # Enrolled speaker profiles
├── frontend/
│   └── index.html           # Web UI
├── START_SERVER.bat         # Quick start script
└── README_USAGE.md          # This file
```

## 🎭 Optional: Named Speakers

To get "Dentist" instead of "Speaker_0":

### Method 1: API (http://localhost:8000/docs)
1. Go to `/enroll` endpoint
2. Upload 15-second audio sample
3. Set `speaker_id` = "Dentist"
4. Repeat for other speakers

### Method 2: Python Script
```powershell
python scripts/enroll_speaker.py --speaker-id Dentist --audio sample.wav
```

## 🔧 Troubleshooting

### Server Won't Start
```powershell
# Check if port 8000 is in use
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue

# Kill process if needed
Get-Process | Where {$_.ProcessName -eq "python"} | Stop-Process
```

### MP3 Files Not Working
Install FFmpeg:
```powershell
choco install ffmpeg -y
```
Then restart server.

### Models Keep Re-downloading
Models are in `pretrained_models/spkrec/`. If deleted, they will re-download (~83MB).

### Only Detects 1 Speaker
- Audio might have very similar voices
- Try enrolling speakers first
- Check if both speakers are clearly audible

## 📊 Technical Details

### Models Used
- **Speaker Recognition**: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
- **Voice Activity Detection**: Silero VAD
- **Speech-to-Text**: SpeechBrain ASR with dental vocabulary
- **Diarization**: Spectral clustering on voice embeddings

### Performance
- **First upload**: 2-5 minutes (model loading)
- **Subsequent uploads**: 10-30 seconds (depending on audio length)
- **Memory**: ~2GB RAM recommended

### Supported Audio Formats
- ✅ WAV (works immediately)
- ✅ FLAC (works immediately)
- ⚠️ MP3 (requires FFmpeg)
- ⚠️ M4A (requires FFmpeg)

## 🎪 System Capabilities

### Current Features (Active)
- [x] Multi-speaker detection
- [x] Speaker diarization  
- [x] Speech-to-text transcription
- [x] Dental terminology support
- [x] Speaker enrollment
- [x] Batch processing
- [x] Web UI

### Disabled Features
- [ ] SOAP note generation (commented out for performance)
- [ ] Real-time WebSocket streaming (available but not tested)
- [ ] LLM integration (commented out)

## 🚨 Known Issues

### Issue: Symlink Errors on Windows
- **Status**: Mitigated
- **Workaround**: Files copied automatically
- **Permanent Fix**: Enable Developer Mode or run as Administrator (not recommended)

### Issue: FFmpeg Missing
- **Status**: Expected
- **Solution**: Install FFmpeg or use WAV files

## 📚 API Documentation

When server is running, visit:
- **Interactive API**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Web UI**: http://localhost:8000

### Key Endpoints
- `POST /process` - Upload audio for processing
- `POST /enroll` - Enroll a speaker
- `GET /health` - Check server status
- `GET /api/info` - System information

## 🎓 How Multi-Speaker ID Works

1. **VAD** (Voice Activity Detection)
   - Detects speech segments
   - Filters out silence and noise

2. **Embedding Extraction**
   - ECAPA-TDNN generates voice "fingerprints"
   - 192-dimensional embedding per segment

3. **Clustering**
   - Groups similar embeddings
   - Each cluster = one speaker

4. **Speaker Matching**
   - Compares to enrolled speakers
   - Assigns names or generic IDs

5. **Transcription**
   - Speech-to-text for each segment
   - Enhanced with dental vocabulary

## ✨ Tips for Best Results

1. **Audio Quality**
   - Clear recording
   - Minimal background noise
   - 16kHz or higher sample rate

2. **Speaker Separation**
   - At least 2-3 seconds per speaker turn
   - Avoid overlapping speech
   - Clear voice differences

3. **Enrollment**
   - Use clean 15-30 second samples
   - Same audio quality as target recordings
   - Multiple samples per speaker improves accuracy

4. **File Length**
   - Minimum: 30 seconds
   - Optimal: 2-10 minutes
   - Maximum: Limited by RAM

## 🎉 Your System is Ready!

Everything is set up and working. Just:
1. Start the server: `START_SERVER.bat`
2. Go to: http://localhost:8000
3. Upload your audio file
4. Get multi-speaker transcriptions!

**Enjoy your voice intelligence system!** 🚀

---

*For technical support or questions, refer to the inline code comments or API documentation at /docs*
