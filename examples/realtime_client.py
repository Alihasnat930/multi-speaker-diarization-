"""
Example client for real-time audio streaming to the dental voice intelligence system.
Demonstrates how to capture microphone audio and stream it to the WebSocket API.
"""
import asyncio
import websockets
import pyaudio
import json
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeAudioClient:
    """
    Client for streaming audio to the dental voice intelligence system.
    Captures microphone input and sends to WebSocket endpoint.
    """
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8000/ws/stream",
        mode: str = "single",
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1
    ):
        """
        Initialize real-time audio client.
        
        Args:
            server_url: WebSocket server URL
            mode: 'single' or 'dual' microphone mode
            sample_rate: Audio sample rate (16000 recommended)
            chunk_size: Audio chunk size in samples
            channels: 1 for mono, 2 for stereo (dual mic)
        """
        self.server_url = f"{server_url}?mode={mode}"
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.mode = mode
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_recording = False
        
    def start_recording(self):
        """Start capturing microphone audio"""
        logger.info(f"Starting audio capture (SR: {self.sample_rate}Hz, Channels: {self.channels})")
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=None
        )
        
        self.is_recording = True
        logger.info("✓ Recording started")
        
    def stop_recording(self):
        """Stop capturing audio"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        logger.info("Recording stopped")
        
    def cleanup(self):
        """Cleanup PyAudio resources"""
        if self.stream:
            self.stop_recording()
        self.audio.terminate()
        
    async def stream_audio(self, duration: Optional[float] = None):
        """
        Stream audio to server via WebSocket.
        
        Args:
            duration: Recording duration in seconds (None = infinite)
        """
        async with websockets.connect(self.server_url) as websocket:
            logger.info(f"Connected to {self.server_url}")
            
            # Wait for connection confirmation
            response = await websocket.recv()
            logger.info(f"Server response: {response}")
            
            # Start recording
            self.start_recording()
            
            # Create tasks for sending and receiving
            send_task = asyncio.create_task(
                self._send_audio_loop(websocket, duration)
            )
            receive_task = asyncio.create_task(
                self._receive_transcript_loop(websocket)
            )
            
            # Wait for tasks
            try:
                await asyncio.gather(send_task, receive_task)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                self.stop_recording()
                
    async def _send_audio_loop(self, websocket, duration: Optional[float]):
        """Send audio chunks to server"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            while self.is_recording:
                if duration and (asyncio.get_event_loop().time() - start_time) > duration:
                    break
                    
                # Read audio chunk
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Send to server
                await websocket.send(audio_data)
                
                # Small delay to avoid overwhelming the server
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Send error: {e}")
            
    async def _receive_transcript_loop(self, websocket):
        """Receive and display transcripts from server"""
        try:
            while True:
                response = await websocket.recv()
                
                try:
                    data = json.loads(response)
                    
                    if data.get('type') == 'transcript':
                        segment = data['data']
                        speaker = segment.get('speaker', 'Unknown')
                        text = segment.get('text', '')
                        start = segment.get('start', 0)
                        end = segment.get('end', 0)
                        
                        # Display transcript
                        print(f"\n[{start:.1f}s - {end:.1f}s] {speaker}: {text}")
                        
                    elif data.get('type') == 'error':
                        logger.error(f"Server error: {data.get('message')}")
                        
                    elif data.get('type') == 'status':
                        logger.info(f"Status: {data.get('message')}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON response: {response}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server")
        except Exception as e:
            logger.error(f"Receive error: {e}")
            
    async def send_command(self, websocket, command: str):
        """Send a command to the server"""
        await websocket.send(json.dumps({
            'type': 'command',
            'command': command
        }))
        
    async def get_history_and_soap(self):
        """Connect, get conversation history and generate SOAP note"""
        async with websockets.connect(self.server_url) as websocket:
            # Get connection confirmation
            await websocket.recv()
            
            # Request history
            logger.info("Requesting conversation history...")
            await self.send_command(websocket, 'history')
            response = await websocket.recv()
            history_data = json.loads(response)
            
            print("\n=== Conversation History ===")
            for segment in history_data.get('data', []):
                print(f"[{segment['start']:.1f}s] {segment['speaker']}: {segment['text']}")
                
            # Request SOAP note
            logger.info("\nGenerating SOAP note...")
            await self.send_command(websocket, 'soap')
            response = await websocket.recv()
            soap_data = json.loads(response)
            
            print("\n=== SOAP Clinical Note ===")
            soap_note = soap_data.get('data', {})
            if 'subjective' in soap_note:
                print(f"\nSUBJECTIVE:\n{soap_note['subjective']}")
                print(f"\nOBJECTIVE:\n{soap_note['objective']}")
                print(f"\nASSESSMENT:\n{soap_note['assessment']}")
                print(f"\nPLAN:\n{soap_note['plan']}")
            else:
                print(soap_note)


class DualMicrophoneClient(RealtimeAudioClient):
    """
    Client for dual microphone setup (e.g., separate dentist and patient mics).
    """
    
    def __init__(self, server_url: str = "ws://localhost:8000/ws/stream", **kwargs):
        super().__init__(
            server_url=server_url,
            mode="dual",
            channels=2,  # Stereo for dual mic
            **kwargs
        )


async def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time dental conversation client')
    parser.add_argument('--mode', choices=['single', 'dual'], default='single',
                       help='Microphone mode')
    parser.add_argument('--duration', type=float, default=None,
                       help='Recording duration in seconds (default: infinite)')
    parser.add_argument('--server', default='ws://localhost:8000/ws/stream',
                       help='WebSocket server URL')
    
    args = parser.parse_args()
    
    # Create client
    if args.mode == 'dual':
        client = DualMicrophoneClient(server_url=args.server)
    else:
        client = RealtimeAudioClient(server_url=args.server, mode=args.mode)
        
    try:
        print(f"\n🎙️  Real-time Dental Voice Intelligence Client")
        print(f"Mode: {args.mode}")
        print(f"Server: {args.server}")
        print(f"\nPress Ctrl+C to stop\n")
        
        # Stream audio
        await client.stream_audio(duration=args.duration)
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
