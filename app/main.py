"""
FastAPI application for dental voice intelligence system.
Supports both batch processing and real-time streaming.
"""
# CRITICAL: Set this BEFORE any imports that might use HuggingFace Hub
import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
import logging
from typing import Optional
from pathlib import Path

from app.diarization import process_audio
# SOAP/LLM generation commented out - focusing on speaker ID and transcription
# from app.summarizer_local import summarize_conversation, get_generator
from app.streaming_api import handle_websocket_connection, stream_manager
from scripts.utils_audio import convert_to_wav_mono_16k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dental Voice Intelligence System",
    description="Real-time voice processing for dental clinics with ASR, diarization, and multi-speaker identification (SOAP/LLM disabled)",
    version="1.0.0-lite"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Redirect to frontend UI"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        # Fallback to API info if frontend not found
        return {
            "service": "Dental Voice Intelligence System",
            "version": "1.0.0",
            "focus": "Multi-Speaker Identification & Transcription",
            "note": "SOAP/LLM features disabled for performance",
            "endpoints": {
                "batch_processing": "/process (POST)",
                "realtime_streaming": "/ws/stream (WebSocket)",
                "health_check": "/health (GET)",
                "speaker_enrollment": "/enroll (POST)",
                "session_history": "/sessions/{id}/history (GET)",
                "api_docs": "/docs (GET)"
            }
        }


@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "service": "Dental Voice Intelligence System",
        "version": "1.0.0-lite",
        "focus": "Multi-Speaker Identification & Transcription",
        "note": "SOAP/LLM features disabled for performance",
        "endpoints": {
            "batch_processing": "/process (POST)",
            "realtime_streaming": "/ws/stream (WebSocket)",
            "health_check": "/health (GET)",
            "speaker_enrollment": "/enroll (POST)",
            "session_history": "/sessions/{id}/history (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dental-voice-engine"}


@app.post('/process')
async def process_batch(file: UploadFile = File(...)):
    """
    Batch process an audio file: diarization + transcription + SOAP note generation.
    
    Args:
        file: Audio file (.wav, .mp3, .m4a)
        
    Returns:
        JSON with transcript segments and SOAP note
    """
    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(
            status_code=400,
            detail='Audio file required (.wav, .mp3, .m4a, .flac)'
        )
        
    # Generate unique temp filename
    file_id = uuid.uuid4().hex[:8]
    tmp_upload = f'tmp_upload_{file_id}_{file.filename}'
    
    try:
        # Save uploaded file
        with open(tmp_upload, 'wb') as f:
            shutil.copyfileobj(file.file, f)
            
        # Convert to proper format
        in_wav = tmp_upload + '.wav'
        logger.info(f"Converting {tmp_upload} to {in_wav}")
        convert_to_wav_mono_16k(tmp_upload, in_wav)
        
        # Process audio (diarization + transcription)
        logger.info(f"Processing audio: {in_wav}")
        results = process_audio(in_wav)
        
        # SOAP note generation commented out - focusing on speaker ID and transcription
        # logger.info("Generating SOAP note...")
        # summary = summarize_conversation(results)
        
        return {
            'segments': results,
            # 'summary': summary,  # Commented out
            'file_id': file_id,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temp files
        try:
            if os.path.exists(tmp_upload):
                os.remove(tmp_upload)
            if os.path.exists(in_wav):
                os.remove(in_wav)
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


@app.post('/enroll')
async def enroll_speaker(
    speaker_id: str = Query(..., description="Speaker ID (e.g., 'Dentist', 'Patient')"),
    file: UploadFile = File(...)
):
    """
    Enroll a speaker for recognition.
    Upload a clean audio sample of the speaker (10-30 seconds recommended).
    
    Args:
        speaker_id: Identifier for the speaker
        file: Audio file with speaker's voice
        
    Returns:
        Enrollment status
    """
    if not file.filename.endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(
            status_code=400,
            detail='Audio file required (.wav, .mp3, .m4a)'
        )
        
    file_id = uuid.uuid4().hex[:8]
    tmp_upload = f'tmp_enroll_{file_id}_{file.filename}'
    
    try:
        # Save file
        with open(tmp_upload, 'wb') as f:
            shutil.copyfileobj(file.file, f)
            
        # Convert to proper format
        in_wav = tmp_upload + '.wav'
        convert_to_wav_mono_16k(tmp_upload, in_wav)
        
        # Enroll speaker
        from app.spk_service import SpeakerService
        spk_service = SpeakerService()
        spk_service.enroll(speaker_id, in_wav)
        
        logger.info(f"Speaker '{speaker_id}' enrolled successfully")
        
        return {
            'status': 'success',
            'speaker_id': speaker_id,
            'message': f'Speaker {speaker_id} enrolled successfully'
        }
        
    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        try:
            if os.path.exists(tmp_upload):
                os.remove(tmp_upload)
            if os.path.exists(in_wav):
                os.remove(in_wav)
        except:
            pass


@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    mode: str = Query("single", description="'single' or 'dual' microphone mode")
):
    """
    WebSocket endpoint for real-time audio streaming.
    
    Protocol:
    1. Connect to ws://host:port/ws/stream?mode=single (or dual)
    2. Send audio chunks as binary data (16-bit PCM, 16kHz, mono or stereo)
    3. Receive JSON responses with transcripts
    4. Send JSON commands: {'type': 'command', 'command': 'history'/'soap'/'reset'}
    
    Args:
        mode: 'single' for single mic, 'dual' for dual microphone (stereo)
    """
    session_id = uuid.uuid4().hex
    logger.info(f"New WebSocket connection: {session_id} (mode: {mode})")
    
    await handle_websocket_connection(websocket, session_id, mode)


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Get conversation history for an active streaming session.
    """
    history = await stream_manager.get_conversation_history(session_id)
    
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return {"session_id": session_id, "history": history}


# SOAP note generation endpoint commented out - focusing on speaker ID and transcription
# @app.post("/sessions/{session_id}/soap")
# async def generate_session_soap(session_id: str):
#     """
#     Generate SOAP note for an active streaming session.
#     """
#     soap_note = await stream_manager.generate_soap_note(session_id)
#     
#     if soap_note is None:
#         raise HTTPException(status_code=404, detail="Session not found")
#         
#     return {"session_id": session_id, "soap_note": soap_note}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Dental Voice Intelligence System...")
    logger.info("Initializing AI models (this may take a few minutes)...")
    
    # Pre-load models (SOAP generator commented out)
    # try:
    #     # Initialize SOAP generator
    #     get_generator()
    #     logger.info("✓ SOAP generator initialized")
    # except Exception as e:
    #     logger.warning(f"SOAP generator initialization delayed: {e}")
        
    logger.info("System ready! (Focus: Speaker ID + Transcription)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    # Cleanup any active sessions
    for session_id in list(stream_manager.engines.keys()):
        stream_manager.disconnect(session_id)
