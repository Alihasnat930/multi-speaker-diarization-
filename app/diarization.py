from app.spk_service import SpeakerService
from app.asr_service import ASRService
from scripts.diarize_cluster import diarize
import tempfile
from scripts.utils_audio import read_audio, write_wav

# Lazy initialization to avoid loading models at import time
_spk = None
_asr = None

def _get_speaker_service():
    global _spk
    if _spk is None:
        _spk = SpeakerService()
    return _spk

def _get_asr_service():
    global _asr
    if _asr is None:
        _asr = ASRService()
    return _asr

def process_audio(audio_path):
    spk = _get_speaker_service()
    asr = _get_asr_service()
    
    dia = diarize(audio_path)
    results = []
    wav, sr = read_audio(audio_path)
    for seg in dia:
        start, end = seg['start'], seg['end']
        cluster_speaker = seg.get('speaker', 0)  # Get cluster label from diarization
        
        s_sample = int(start*sr); e_sample = int(end*sr)
        tmp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        write_wav(tmp_path, wav[s_sample:e_sample], sr)
        text = asr.transcribe_file(tmp_path)
        
        # Get embedding for potential matching with enrolled speakers
        emb = spk.get_embedding_file(tmp_path)
        enrolled_id, enrolled_score = spk.match_embedding(emb)
        
        # Use enrolled speaker if match is good (score > 0.7), otherwise use cluster label
        if enrolled_id and enrolled_score > 0.7:
            speaker_id = enrolled_id
            score = enrolled_score
        else:
            speaker_id = f"Speaker_{cluster_speaker}"
            # Calculate a confidence score based on clustering
            # Since we don't have a direct score, use a placeholder
            score = 0.0  # Will be updated with proper score later
        
        results.append({
            'start': start, 
            'end': end, 
            'speaker_id': speaker_id, 
            'score': float(score), 
            'text': text
        })
    return results
