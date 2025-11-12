"""
ASR Service with support for fine-tuned dental terminology models.
Uses SpeechBrain's ASR models with optional custom vocabulary.
"""
from speechbrain.pretrained import EncoderDecoderASR, EncoderASR
import torch
import logging
import os
from pathlib import Path
from typing import Optional, Union

# Disable symlinks on Windows to avoid privilege issues
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASR_DIR = 'pretrained_models/asr'
DENTAL_VOCAB_PATH = 'models/dental_vocabulary.txt'


class ASRService:
    """
    Automatic Speech Recognition service optimized for dental clinical conversations.
    Supports fine-tuned models with dental terminology.
    """
    
    def __init__(
        self,
        model_source: str = 'speechbrain/asr-crdnn-rnnlm-librispeech',
        use_dental_lm: bool = True
    ):
        """
        Initialize ASR service.
        
        Args:
            model_source: HuggingFace model ID or local path to fine-tuned model
            use_dental_lm: Whether to use dental language model adaptation
        """
        logger.info(f"Loading Whisper ASR model (Windows-compatible)")
        
        # Use Whisper for transcription - works reliably on Windows
        try:
            import whisper
            self.asr = whisper.load_model("base")  # Options: tiny, base, small, medium, large
            self.use_whisper = True
            logger.info("✓ Whisper ASR loaded successfully (base model)")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self.asr = None
            self.use_whisper = False
            logger.warning("ASR disabled - will show placeholder text")
        
        # Load dental vocabulary if available
        self.dental_vocab = self._load_dental_vocabulary() if use_dental_lm else None
        
        logger.info("ASR service initialized")
        
    def _load_dental_vocabulary(self) -> Optional[set]:
        """Load dental-specific vocabulary for post-processing"""
        vocab_path = Path(DENTAL_VOCAB_PATH)
        if vocab_path.exists():
            logger.info(f"Loading dental vocabulary from {vocab_path}")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = set(line.strip().lower() for line in f if line.strip())
            logger.info(f"Loaded {len(vocab)} dental terms")
            return vocab
        else:
            logger.warning(f"Dental vocabulary not found at {vocab_path}")
            return None
            
    def transcribe_file(self, wav_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            wav_path: Path to WAV file (16kHz, mono recommended)
            
        Returns:
            Transcribed text
        """
        if self.asr is None:
            # Fallback if ASR not loaded
            import soundfile as sf
            try:
                audio, sr = sf.read(wav_path)
                duration = len(audio) / sr
                return f"[Speaker segment: {duration:.1f}s]"
            except:
                return "[Speaker speaking]"
        
        if self.use_whisper:
            # Use Whisper for transcription
            try:
                import os
                import soundfile as sf
                import numpy as np
                
                # Ensure the file exists
                if not os.path.exists(wav_path):
                    logger.error(f"Audio file not found: {wav_path}")
                    return "[File not found]"
                
                # Load audio with soundfile (avoids FFmpeg dependency)
                audio, sr = sf.read(wav_path)
                
                # Convert to float32 mono if needed
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
                
                # Resample to 16kHz if needed (Whisper expects 16kHz)
                if sr != 16000:
                    from scipy import signal
                    num_samples = int(len(audio) * 16000 / sr)
                    audio = signal.resample(audio, num_samples)
                
                # Transcribe with Whisper using audio array directly
                result = self.asr.transcribe(
                    audio,  # Pass audio array instead of file path
                    language="en",
                    fp16=False,  # Disable FP16 on CPU
                    verbose=False
                )
                text = result["text"].strip()
                
                # Post-process with dental vocabulary if available
                if self.dental_vocab and text:
                    text = self._apply_dental_corrections(text)
                
                return text if text else "[No speech detected]"
            except FileNotFoundError as e:
                logger.error(f"File not found for transcription: {wav_path}")
                return "[File not found]"
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                import traceback
                traceback.print_exc()
                return "[Transcription error]"
            
    def transcribe_batch(self, wav_paths: list) -> list:
        """
        Transcribe multiple audio files in batch.
        
        Args:
            wav_paths: List of paths to WAV files
            
        Returns:
            List of transcribed texts
        """
        results = []
        for path in wav_paths:
            text = self.transcribe_file(path)
            results.append(text)
        return results
        
    def _apply_dental_corrections(self, text: str) -> str:
        """
        Apply dental terminology corrections to transcription.
        This is a simple implementation - for production, use more sophisticated methods.
        """
        # Common dental term corrections
        corrections = {
            'cavity': ['cavities', 'cavity'],
            'molar': ['molar', 'molars'],
            'incisor': ['incisor', 'incisors'],
            'canine': ['canine', 'canines'],
            'premolar': ['premolar', 'premolars'],
            'gingiva': ['gingiva', 'gingival'],
            'periodontal': ['periodontal', 'periodontitis'],
            'extraction': ['extraction', 'extract'],
            'crown': ['crown', 'crowns'],
            'bridge': ['bridge', 'bridges'],
            'implant': ['implant', 'implants'],
            'root canal': ['root canal'],
            'filling': ['filling', 'fillings'],
            'plaque': ['plaque'],
            'tartar': ['tartar', 'calculus'],
            'enamel': ['enamel'],
            'dentin': ['dentin', 'dentine'],
            'pulp': ['pulp'],
            'abscess': ['abscess'],
            'gingivitis': ['gingivitis'],
            'orthodontic': ['orthodontic', 'orthodontics'],
            'braces': ['braces'],
            'retainer': ['retainer', 'retainers'],
            'x-ray': ['x-ray', 'xray', 'radiograph'],
            'anesthesia': ['anesthesia', 'anesthetic'],
            'novocaine': ['novocaine'],
            'fluoride': ['fluoride'],
            'sealant': ['sealant', 'sealants'],
        }
        
        # Simple word-level correction (for production, use phonetic matching)
        words = text.split()
        corrected_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?')
            # Check if word should be corrected
            # This is a placeholder - implement proper correction logic
            corrected_words.append(word)
            
        return ' '.join(corrected_words)


class StreamingASRService(ASRService):
    """
    Streaming ASR service for real-time transcription.
    Uses chunked processing with context carryover.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context_window = []
        self.max_context_chunks = 3
        
    def transcribe_streaming_chunk(
        self,
        audio_chunk: torch.Tensor,
        carry_context: bool = True
    ) -> str:
        """
        Transcribe streaming audio chunk with context.
        
        Args:
            audio_chunk: Audio tensor
            carry_context: Whether to use previous chunks as context
            
        Returns:
            Transcribed text for this chunk
        """
        # This is a placeholder for streaming implementation
        # Full streaming ASR requires model-specific implementation
        # For now, fall back to file-based transcription
        
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            if isinstance(audio_chunk, torch.Tensor):
                audio_np = audio_chunk.cpu().numpy()
            else:
                audio_np = audio_chunk
                
            sf.write(tmp.name, audio_np, 16000)
            text = self.transcribe_file(tmp.name)
            
        return text
