# Multi-Speaker Identification Guide

## 🎯 How It Works

Your system is **already configured for multi-speaker identification**. Here's what happens when you upload a conversation:

### 1. **Voice Activity Detection (VAD)**
   - Silero VAD model detects when people are speaking
   - Filters out silence, noise, and non-speech sounds
   - Identifies speech segments with precise timestamps

### 2. **Speaker Diarization** 
   - Separates audio into segments by different speakers
   - Uses spectral clustering on voice embeddings
   - Answers: "Who spoke when?"

### 3. **Speaker Recognition (Identification)**
   - ECAPA-TDNN model extracts voice embeddings (speaker fingerprints)
   - Compares embeddings to enrolled speakers
   - Assigns speaker IDs to each segment

### 4. **Speech-to-Text Transcription**
   - SpeechBrain ASR transcribes each speaker segment
   - Enhanced with dental terminology vocabulary
   - Produces text for each speaker turn

## 📊 Example Output

When you process a 2-person conversation, you'll get:

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
      "text": "I have some pain in my upper molar"
    },
    {
      "start": 7.0,
      "end": 10.5,
      "speaker_id": "Speaker_0",
      "score": 0.88,
      "text": "Let me examine that area"
    }
  ]
}
```

## 🎪 Current Issue: FFmpeg Missing

**Problem:** MP3/M4A files can't be processed without FFmpeg

**Solutions:**

### Option A: Install FFmpeg (Recommended)
See `INSTALL_FFMPEG.md` for detailed instructions:

**Quick install with Chocolatey:**
```powershell
choco install ffmpeg -y
```

**Or download manually:**
1. https://www.gyan.dev/ffmpeg/builds/
2. Extract to C:\ffmpeg
3. Add C:\ffmpeg\bin to PATH
4. Restart terminal

### Option B: Use WAV Files
- Convert your MP3 to WAV using online tools
- Upload WAV files directly
- System works perfectly with WAV files

## 🎭 Improving Speaker Identification

Right now, speakers are labeled as "Speaker_0", "Speaker_1", etc.

### To Get Named Speakers (e.g., "Dentist", "Patient"):

1. **Enroll Speakers** (one-time setup)
   
   Go to http://localhost:8000/docs and find `/enroll` endpoint:
   
   - Upload a 10-30 second audio sample of the dentist's voice
   - Set `speaker_id` = "Dentist"
   - Upload a sample of the patient's voice
   - Set `speaker_id` = "Patient"

2. **Or Use Python Script:**
   ```powershell
   python scripts/enroll_speaker.py --speaker-id Dentist --audio dentist_sample.wav
   python scripts/enroll_speaker.py --speaker-id Patient --audio patient_sample.wav
   ```

3. **Process New Audio:**
   - After enrollment, the system will match voices
   - Output will show "Dentist" and "Patient" instead of generic IDs
   - Matching confidence score indicates reliability

## 🧪 Testing Multi-Speaker Identification

### Test with your own audio:
```powershell
python test_speaker_id.py your_conversation.wav
```

### What to expect:
- ✅ Number of speakers detected
- ✅ Speaker IDs for each segment
- ✅ Timestamps and transcriptions
- ✅ Confidence scores (0-1)

## 📝 Tips for Best Results

1. **Audio Quality:**
   - Clear recording, minimal background noise
   - 16kHz or higher sample rate
   - Each speaker should speak for at least 2-3 seconds per turn

2. **File Formats:**
   - WAV: ✅ Works immediately
   - FLAC: ✅ Works immediately
   - MP3: ⚠️ Needs FFmpeg
   - M4A: ⚠️ Needs FFmpeg

3. **Speaker Enrollment:**
   - Use clean samples (no overlapping speech)
   - 10-30 seconds per speaker is ideal
   - More samples = better accuracy

4. **Conversation Length:**
   - Minimum: 30 seconds (for meaningful diarization)
   - Optimal: 2-10 minutes
   - Maximum: Limited by system memory

## 🔧 Current Status

| Feature | Status |
|---------|--------|
| Multi-speaker detection | ✅ Working |
| Voice activity detection | ✅ Working |
| Speaker diarization | ✅ Working |
| Speaker identification | ✅ Working |
| Speech-to-text | ✅ Working |
| Dental terminology | ✅ Working |
| Speaker enrollment | ✅ Working |
| MP3/M4A support | ⚠️ Needs FFmpeg |
| WAV/FLAC support | ✅ Working |
| SOAP note generation | ⏸️ Disabled |

## 🚀 Quick Start

1. **Install FFmpeg** (if you have MP3/M4A files):
   ```powershell
   choco install ffmpeg -y
   ```

2. **Restart the server** (in the server terminal window):
   - Press Ctrl+C
   - Run: `START_SERVER.bat`

3. **Upload your audio file** at http://localhost:8000

4. **View results:**
   - See all speaker segments
   - Each with speaker ID, timestamp, and transcription
   - Multiple speakers automatically identified

## 🆘 Troubleshooting

**"System cannot find file specified"**
- Install FFmpeg for MP3/M4A files
- Or use WAV format instead

**"Only detecting 1 speaker"**
- Audio might have very similar voices
- Try enrolling speakers first
- Check if both speakers are speaking clearly

**"Speaker IDs keep changing"**
- Enroll speakers for consistent naming
- Without enrollment, IDs are assigned dynamically

**"Low confidence scores"**
- Audio quality might be poor
- Background noise affecting recognition
- Try enrolling with cleaner samples

## 📚 Next Steps

1. Install FFmpeg following `INSTALL_FFMPEG.md`
2. Restart your server
3. Upload a multi-person conversation
4. See automatic speaker separation in action!
