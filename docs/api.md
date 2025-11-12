# API Documentation

Complete REST and WebSocket API reference for the Dental Voice Intelligence System.

## Base URL

```
http://localhost:8000
```

## Authentication

Current version: No authentication (for internal clinic network)

For production deployment, add API key authentication or integrate with your EHR system.

---

## REST API Endpoints

### 1. Health Check

Check if the service is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "dental-voice-engine"
}
```

---

### 2. API Information

Get API version and available endpoints.

**Endpoint:** `GET /`

**Response:**
```json
{
  "service": "Dental Voice Intelligence System",
  "version": "1.0.0",
  "endpoints": {
    "batch_processing": "/process (POST)",
    "realtime_streaming": "/ws/stream (WebSocket)",
    "health_check": "/health (GET)",
    "speaker_enrollment": "/enroll (POST)"
  }
}
```

---

### 3. Process Audio File (Batch)

Upload and process a pre-recorded audio file. Returns transcript segments and SOAP note.

**Endpoint:** `POST /process`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (file, required): Audio file (.wav, .mp3, .m4a, .flac)

**Request Example (cURL):**
```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@consultation_recording.mp3"
```

**Request Example (Python):**
```python
import requests

with open('consultation.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process',
        files={'file': f}
    )
    result = response.json()
```

**Response (200 OK):**
```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker_id": "Dentist",
      "score": 0.92,
      "text": "How are you feeling today?"
    },
    {
      "start": 3.8,
      "end": 6.1,
      "speaker_id": "Patient",
      "score": 0.88,
      "text": "I've been having some pain in my upper left molar"
    }
  ],
  "summary": "SOAP Clinical Note\n==================\n\nSUBJECTIVE:\nPatient reports pain in upper left molar...\n\nOBJECTIVE:\nExamination reveals...\n\nASSESSMENT:\nDental caries, tooth #14...\n\nPLAN:\nComposite filling recommended...",
  "file_id": "a3b8c9d2",
  "status": "success"
}
```

**Error Response (400 Bad Request):**
```json
{
  "detail": "Audio file required (.wav, .mp3, .m4a, .flac)"
}
```

**Error Response (500 Internal Server Error):**
```json
{
  "detail": "Error message describing the issue"
}
```

---

### 4. Enroll Speaker

Enroll a speaker (dentist or patient) for voice identification.

**Endpoint:** `POST /enroll`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `speaker_id` (query string, required): Speaker identifier (e.g., "Dentist", "Patient", "Dr_Smith")
- `file` (file, required): Audio sample of speaker's voice (10-30 seconds recommended)

**Request Example (cURL):**
```bash
curl -X POST "http://localhost:8000/enroll?speaker_id=Dentist" \
  -F "file=@dentist_sample.wav"
```

**Request Example (Python):**
```python
import requests

with open('dentist_voice.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/enroll',
        params={'speaker_id': 'Dentist'},
        files={'file': f}
    )
```

**Response (200 OK):**
```json
{
  "status": "success",
  "speaker_id": "Dentist",
  "message": "Speaker Dentist enrolled successfully"
}
```

---

### 5. Get Session History

Retrieve conversation history for an active streaming session.

**Endpoint:** `GET /sessions/{session_id}/history`

**Parameters:**
- `session_id` (path, required): Session identifier from WebSocket connection

**Response (200 OK):**
```json
{
  "session_id": "a3b8c9d2e5f6g7h8",
  "history": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker": "Dentist",
      "text": "How are you feeling?",
      "confidence": 1.0,
      "speaker_confidence": 0.95
    }
  ]
}
```

---

### 6. Generate SOAP Note for Session

Generate SOAP note from an active streaming session's conversation history.

**Endpoint:** `POST /sessions/{session_id}/soap`

**Response (200 OK):**
```json
{
  "session_id": "a3b8c9d2e5f6g7h8",
  "soap_note": {
    "subjective": "Patient reports...",
    "objective": "Examination reveals...",
    "assessment": "Diagnosis...",
    "plan": "Treatment plan..."
  }
}
```

---

## WebSocket API

### Real-Time Audio Streaming

Stream live audio for real-time transcription and speaker identification.

**Endpoint:** `ws://localhost:8000/ws/stream`

**Query Parameters:**
- `mode` (optional): `single` (default) or `dual` for dual microphone setup

**Connection URL:**
```
ws://localhost:8000/ws/stream?mode=single
```

---

### WebSocket Protocol

#### 1. Connection

Client connects to WebSocket endpoint. Server responds with connection confirmation:

**Server → Client (on connect):**
```json
{
  "type": "connection",
  "status": "connected",
  "session_id": "a3b8c9d2e5f6g7h8",
  "mode": "single"
}
```

---

#### 2. Audio Streaming

Client sends binary audio data continuously.

**Client → Server (binary):**
- Format: 16-bit PCM audio
- Sample rate: 16kHz (16000 Hz)
- Channels: 1 (mono) for single mode, 2 (stereo) for dual mode
- Chunk size: 1024-8192 samples recommended

**Python Example:**
```python
import websockets
import pyaudio

async with websockets.connect('ws://localhost:8000/ws/stream?mode=single') as ws:
    # Receive connection confirmation
    response = await ws.recv()
    
    # Setup audio capture
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )
    
    # Stream audio
    while True:
        audio_chunk = stream.read(1024)
        await ws.send(audio_chunk)
```

---

#### 3. Transcript Reception

When a speech segment is detected and transcribed, server sends transcript:

**Server → Client (JSON):**
```json
{
  "type": "transcript",
  "data": {
    "start": 12.5,
    "end": 15.8,
    "speaker": "Dentist",
    "text": "Let me examine your tooth",
    "confidence": 1.0,
    "speaker_confidence": 0.92
  },
  "timestamp": "2025-11-06T14:32:45.123456"
}
```

---

#### 4. Commands

Client can send JSON commands to the server:

**Get Conversation History:**

**Client → Server:**
```json
{
  "type": "command",
  "command": "history"
}
```

**Server → Client:**
```json
{
  "type": "history",
  "data": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker": "Dentist",
      "text": "How are you feeling?",
      "confidence": 1.0,
      "speaker_confidence": 0.95
    }
  ]
}
```

---

**Generate SOAP Note:**

**Client → Server:**
```json
{
  "type": "command",
  "command": "soap"
}
```

**Server → Client:**
```json
{
  "type": "soap",
  "data": {
    "subjective": "Patient reports...",
    "objective": "Examination reveals...",
    "assessment": "Diagnosis...",
    "plan": "Treatment plan..."
  }
}
```

---

**Reset Session:**

**Client → Server:**
```json
{
  "type": "command",
  "command": "reset"
}
```

**Server → Client:**
```json
{
  "type": "status",
  "message": "Session reset"
}
```

---

#### 5. Error Messages

**Server → Client (on error):**
```json
{
  "type": "error",
  "message": "Error description"
}
```

---

## Data Models

### TranscriptSegment

```typescript
{
  start: number,              // Start time in seconds
  end: number,                // End time in seconds
  speaker: string,            // Speaker ID ("Dentist", "Patient", etc.)
  speaker_id?: string,        // Alias for speaker
  text: string,               // Transcribed text
  confidence: number,         // ASR confidence (0.0-1.0)
  speaker_confidence: number, // Speaker ID confidence (0.0-1.0)
  score?: number             // Alias for speaker_confidence
}
```

### SOAPNote

```typescript
{
  subjective: string,    // Patient's complaint, symptoms, history
  objective: string,     // Clinical findings, observations
  assessment: string,    // Diagnosis, clinical impression
  plan: string          // Treatment plan, next steps
}
```

---

## Rate Limits

Current version: No rate limits (for internal use)

For production deployment, consider:
- Max 10 concurrent WebSocket connections per clinic
- Max 100 batch processing requests per hour
- Max audio file size: 100MB

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found (session/resource doesn't exist) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model loading) |

---

## Interactive Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Client Libraries

### Python

```python
# Install
pip install requests websockets

# Batch processing
import requests
response = requests.post('http://localhost:8000/process', files={'file': open('audio.wav', 'rb')})

# Real-time streaming
import asyncio
import websockets

async def stream():
    async with websockets.connect('ws://localhost:8000/ws/stream') as ws:
        await ws.recv()  # Connection confirmation
        # Send audio chunks...
        await ws.send(audio_bytes)
        # Receive transcripts...
        response = await ws.recv()

asyncio.run(stream())
```

### JavaScript

```javascript
// Batch processing
const formData = new FormData();
formData.append('file', audioFile);

fetch('http://localhost:8000/process', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Real-time streaming
const ws = new WebSocket('ws://localhost:8000/ws/stream?mode=single');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'transcript') {
    console.log(`${data.data.speaker}: ${data.data.text}`);
  }
};

// Send audio
ws.send(audioBuffer);
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Process audio
curl -X POST http://localhost:8000/process \
  -F "file=@recording.mp3"

# Enroll speaker
curl -X POST "http://localhost:8000/enroll?speaker_id=Dentist" \
  -F "file=@dentist_voice.wav"
```

---

## Best Practices

1. **Audio Quality**
   - Use 16kHz sample rate (higher is okay, will be resampled)
   - Mono for single mic, stereo for dual mic
   - Minimize background noise
   - Ensure speaker is close to microphone

2. **Speaker Enrollment**
   - Use 10-30 seconds of clean speech
   - Avoid background noise
   - Multiple samples per speaker improve accuracy

3. **Real-Time Streaming**
   - Send audio chunks of 1024-2048 samples (64-128ms)
   - Don't send chunks too frequently (overwhelms server)
   - Handle reconnection on disconnect

4. **Error Handling**
   - Always check response status codes
   - Implement retry logic for network errors
   - Log errors for debugging

5. **Security**
   - Use HTTPS/WSS in production
   - Implement authentication
   - Encrypt PHI data at rest
   - Use VPN or private network

---

## Performance Tuning

- **Batch processing**: 5-10 seconds per minute of audio (GPU)
- **Real-time streaming**: Sub-second latency with GPU
- **Concurrent streams**: Up to 4 simultaneous on RTX 3060
- **SOAP generation**: 5-15 seconds depending on conversation length

---

## Support

For API issues or feature requests, contact: api-support@yourorg.com
