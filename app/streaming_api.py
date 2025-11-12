"""
WebSocket-based real-time audio streaming endpoint.
Supports dual microphone input and real-time transcription.
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import asyncio
import numpy as np
import json
import logging
import struct
from datetime import datetime

from app.realtime_engine import RealtimeVoiceEngine, DualMicrophoneEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioStreamManager:
    """
    Manages WebSocket connections for real-time audio streaming.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.engines: Dict[str, RealtimeVoiceEngine] = {}
        
    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        mode: str = "single"
    ):
        """
        Accept WebSocket connection and initialize processing engine.
        
        Args:
            websocket: FastAPI WebSocket connection
            session_id: Unique session identifier
            mode: 'single' for single mic, 'dual' for dual microphone setup
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        # Initialize appropriate engine
        if mode == "dual":
            engine = DualMicrophoneEngine(
                sample_rate=16000,
                vad_threshold=0.5,
                min_speech_duration=0.3,
                min_silence_duration=0.5
            )
        else:
            engine = RealtimeVoiceEngine(
                sample_rate=16000,
                vad_threshold=0.5,
                min_speech_duration=0.3,
                min_silence_duration=0.5
            )
            
        self.engines[session_id] = engine
        
        logger.info(f"Session {session_id} connected ({mode} mode)")
        
        # Send connection confirmation
        await websocket.send_json({
            'type': 'connection',
            'status': 'connected',
            'session_id': session_id,
            'mode': mode
        })
        
    def disconnect(self, session_id: str):
        """Disconnect session and cleanup"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.engines:
            del self.engines[session_id]
        logger.info(f"Session {session_id} disconnected")
        
    async def process_audio_data(
        self,
        session_id: str,
        audio_data: bytes,
        mode: str = "single"
    ):
        """
        Process incoming audio data and return transcript if available.
        
        Args:
            session_id: Session identifier
            audio_data: Raw audio bytes (16-bit PCM, mono or stereo)
            mode: Processing mode
        """
        if session_id not in self.engines:
            logger.error(f"Session {session_id} not found")
            return
            
        engine = self.engines[session_id]
        websocket = self.active_connections[session_id]
        
        try:
            # Convert bytes to numpy array
            # Assuming 16-bit PCM audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1, 1]
            audio_array = audio_array / 32768.0
            
            # Process based on mode
            if mode == "dual" and len(audio_array) % 2 == 0:
                # Stereo: split into left and right channels
                left_channel = audio_array[0::2]
                right_channel = audio_array[1::2]
                
                # Process both channels
                segments = await engine.process_stereo_chunk(left_channel, right_channel)
                
                # Send results for each segment
                for segment in segments:
                    await websocket.send_json({
                        'type': 'transcript',
                        'data': segment.to_dict(),
                        'timestamp': datetime.now().isoformat()
                    })
                    
            else:
                # Single channel
                segment = await engine.process_audio_chunk(audio_array)
                
                if segment is not None:
                    await websocket.send_json({
                        'type': 'transcript',
                        'data': segment.to_dict(),
                        'timestamp': datetime.now().isoformat()
                    })
                    
        except Exception as e:
            logger.error(f"Error processing audio for session {session_id}: {e}")
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
            
    async def get_conversation_history(self, session_id: str) -> Optional[list]:
        """Get conversation history for session"""
        if session_id in self.engines:
            return self.engines[session_id].get_conversation_history()
        return None
        
    # SOAP note generation commented out - focusing on speaker ID and transcription
    # async def generate_soap_note(self, session_id: str) -> Optional[dict]:
    #     """Generate SOAP note for session"""
    #     if session_id not in self.engines:
    #         return None
    #         
    #     engine = self.engines[session_id]
    #     segments = engine.get_conversation_history()
    #     
    #     if not segments:
    #         return {'error': 'No conversation data available'}
    #         
    #     # Generate SOAP note
    #     from app.summarizer_local import get_generator
    #     
    #     try:
    #         generator = get_generator()
    #         if hasattr(generator, 'generate_soap_note'):
    #             soap_note = generator.generate_soap_note(segments)
    #         else:
    #             # Fallback
    #             from app.summarizer_local import summarize_conversation
    #             soap_text = summarize_conversation(segments)
    #             soap_note = {'raw': soap_text}
    #             
    #         return soap_note
    #         
    #     except Exception as e:
    #         logger.error(f"SOAP generation failed: {e}")
    #         return {'error': str(e)}


# Global manager instance
stream_manager = AudioStreamManager()


async def handle_websocket_connection(
    websocket: WebSocket,
    session_id: str,
    mode: str = "single"
):
    """
    Main WebSocket handler for audio streaming.
    
    Protocol:
    - Client sends JSON with {'type': 'config', ...} for configuration
    - Client sends binary data for audio chunks
    - Client sends JSON with {'type': 'command', 'command': 'history'/'soap'/'reset'}
    - Server sends JSON with {'type': 'transcript'/'soap'/'error', ...}
    """
    await stream_manager.connect(websocket, session_id, mode)
    
    try:
        while True:
            # Receive data (can be JSON or binary)
            try:
                # Try to receive as text (JSON)
                message = await websocket.receive_text()
                data = json.loads(message)
                
                # Handle commands
                if data.get('type') == 'command':
                    command = data.get('command')
                    
                    if command == 'history':
                        history = await stream_manager.get_conversation_history(session_id)
                        await websocket.send_json({
                            'type': 'history',
                            'data': history
                        })
                        
                    # SOAP generation commented out
                    # elif command == 'soap':
                    #     soap_note = await stream_manager.generate_soap_note(session_id)
                    #     await websocket.send_json({
                    #         'type': 'soap',
                    #         'data': soap_note
                    #     })
                        
                    elif command == 'reset':
                        if session_id in stream_manager.engines:
                            stream_manager.engines[session_id].reset()
                        await websocket.send_json({
                            'type': 'status',
                            'message': 'Session reset'
                        })
                        
                    else:
                        await websocket.send_json({
                            'type': 'error',
                            'message': f'Unknown command: {command}'
                        })
                        
            except json.JSONDecodeError:
                # Receive as binary (audio data)
                audio_bytes = await websocket.receive_bytes()
                await stream_manager.process_audio_data(session_id, audio_bytes, mode)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        stream_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        stream_manager.disconnect(session_id)
