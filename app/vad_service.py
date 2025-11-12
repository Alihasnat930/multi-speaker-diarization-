"""
Voice Activity Detection (VAD) Service using Silero VAD
High-quality, real-time VAD for detecting speech segments
"""
import torch
import numpy as np
from typing import Union, List, Tuple
import logging

logger = logging.getLogger(__name__)


class VADService:
    """
    Voice Activity Detection using Silero VAD model.
    Optimized for real-time streaming audio.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize VAD service.
        
        Args:
            sample_rate: Audio sample rate (8000 or 16000)
        """
        if sample_rate not in [8000, 16000]:
            raise ValueError("Sample rate must be 8000 or 16000")
            
        self.sample_rate = sample_rate
        
        # Load Silero VAD model
        logger.info("Loading Silero VAD model...")
        self.model, self.utils = self._load_silero_vad()
        
        # Extract utility functions
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils
        
        # Create iterator for streaming
        self.vad_iterator = self.VADIterator(
            model=self.model,
            threshold=0.5,
            sampling_rate=sample_rate,
            min_silence_duration_ms=500,
            speech_pad_ms=30
        )
        
        logger.info("VAD service initialized")
        
    def _load_silero_vad(self):
        """Load Silero VAD model from torch hub"""
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            return model, utils
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise
            
    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Process a single audio chunk and return speech probability.
        
        Args:
            audio_chunk: Audio samples (mono, 16kHz or 8kHz)
            
        Returns:
            Speech probability (0.0 to 1.0)
        """
        # Convert to torch tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        else:
            audio_tensor = audio_chunk
            
        # Ensure proper shape
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
        return speech_prob
        
    def process_chunk_streaming(self, audio_chunk: np.ndarray) -> dict:
        """
        Process chunk using VAD iterator for streaming mode.
        Returns speech dict if speech segment detected.
        
        Args:
            audio_chunk: Audio samples
            
        Returns:
            Dictionary with 'start' and 'end' if speech detected, else None
        """
        # Convert to torch tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        else:
            audio_tensor = audio_chunk
            
        # Process with iterator
        speech_dict = self.vad_iterator(audio_tensor, return_seconds=True)
        
        return speech_dict
        
    def get_speech_segments(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        return_seconds: bool = True
    ) -> List[dict]:
        """
        Get all speech segments from audio file or array.
        
        Args:
            audio: Audio samples or path to file
            return_seconds: Return timestamps in seconds (True) or samples (False)
            
        Returns:
            List of dictionaries with 'start' and 'end' keys
        """
        if isinstance(audio, str):
            # Load from file
            audio = self.read_audio(audio, sampling_rate=self.sample_rate)
        elif isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))
            
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=self.sample_rate,
            return_seconds=return_seconds
        )
        
        return speech_timestamps
        
    def reset_state(self):
        """Reset VAD iterator state (useful for new audio streams)"""
        self.vad_iterator.reset_states()
        logger.debug("VAD state reset")


class EnergyVAD:
    """
    Simple energy-based VAD as fallback.
    Less accurate than Silero but faster and no model dependency.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.02
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_length = int(sample_rate * frame_duration_ms / 1000)
        self.energy_threshold = energy_threshold
        
        # Running statistics for adaptive threshold
        self.energy_history = []
        self.max_history = 100
        
    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Process audio chunk with energy-based VAD.
        
        Returns:
            Pseudo-probability (0.0 or 1.0)
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Update history
        self.energy_history.append(energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
            
        # Adaptive threshold
        if len(self.energy_history) > 10:
            mean_energy = np.mean(self.energy_history)
            threshold = mean_energy * 1.5
        else:
            threshold = self.energy_threshold
            
        # Return binary decision as probability
        return 1.0 if energy > threshold else 0.0
        
    def get_speech_segments(
        self,
        audio: np.ndarray,
        return_seconds: bool = True
    ) -> List[dict]:
        """
        Get speech segments using energy-based detection.
        """
        segments = []
        n_frames = len(audio) // self.frame_length
        
        is_speech = False
        start_frame = None
        
        for i in range(n_frames):
            frame_start = i * self.frame_length
            frame_end = frame_start + self.frame_length
            frame = audio[frame_start:frame_end]
            
            speech_prob = self.process_chunk(frame)
            
            if speech_prob > 0.5 and not is_speech:
                # Speech started
                is_speech = True
                start_frame = i
            elif speech_prob <= 0.5 and is_speech:
                # Speech ended
                is_speech = False
                if return_seconds:
                    segments.append({
                        'start': start_frame * self.frame_length / self.sample_rate,
                        'end': i * self.frame_length / self.sample_rate
                    })
                else:
                    segments.append({
                        'start': start_frame * self.frame_length,
                        'end': i * self.frame_length
                    })
                    
        # Handle final segment
        if is_speech and start_frame is not None:
            if return_seconds:
                segments.append({
                    'start': start_frame * self.frame_length / self.sample_rate,
                    'end': len(audio) / self.sample_rate
                })
            else:
                segments.append({
                    'start': start_frame * self.frame_length,
                    'end': len(audio)
                })
                
        return segments
