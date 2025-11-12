import soundfile as sf
import numpy as np
from scipy import signal
import os

def convert_to_wav_mono_16k(src_path, dst_path):
    """
    Convert audio file to 16kHz mono WAV format.
    Supports WAV, MP3, FLAC, M4A formats.
    """
    # First try with soundfile (works for WAV, FLAC)
    try:
        wav, sr = sf.read(src_path)
        
        # Convert stereo to mono if needed
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            num_samples = int(len(wav) * 16000 / sr)
            wav = signal.resample(wav, num_samples)
        
        # Save as WAV
        sf.write(dst_path, wav, 16000)
        return
        
    except Exception as e:
        # If soundfile fails, try pydub (needs FFmpeg for MP3/M4A)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(src_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(dst_path, format='wav')
            return
        except FileNotFoundError:
            raise RuntimeError(
                f"Cannot convert {src_path} to WAV. "
                "For MP3/M4A files, please install FFmpeg: "
                "https://ffmpeg.org/download.html"
            )
        except Exception as e2:
            raise RuntimeError(f"Audio conversion failed: {e2}")

def read_audio(path):
    wav, sr = sf.read(path)
    return wav, sr

def write_wav(path, samples, sr=16000):
    sf.write(path, samples, sr)
